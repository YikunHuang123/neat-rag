"""
Evaluation Runner

Evaluates the RAG pipeline using ragas metrics:
  faithfulness, answer_relevancy, context_precision

Usage:
    # Run against a JSON file of questions (+ optional ground truths)
    python -m neat_rag.eval --questions eval/questions.json --output report.json

    # Use built-in demo questions (requires indexed documents)
    python -m neat_rag.eval --demo

Evaluation LLM (The "Judge"):
    By default, ragas uses OpenAI (gpt-4o-mini). To use other providers:
    
    1. DeepSeek:
       export OPENAI_API_KEY="your-deepseek-key"
       export OPENAI_API_BASE="https://api.deepseek.com/v1"
       python -m neat_rag.eval --demo --eval-llm deepseek-chat

    2. Gemini (OpenAI-compatible):
       export OPENAI_API_KEY="your-gemini-key"
       export OPENAI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"
       python -m neat_rag.eval --demo --eval-llm gemini-1.5-flash

Input JSON formats accepted:
    ["question 1", "question 2", ...]
    [{"question": "...", "ground_truth": "..."}, ...]

Output (report.json):
    {
      "samples": 10,
      "aggregate": {"faithfulness": 0.82, "answer_relevancy": 0.91, ...},
      "per_sample": [...],
      "config": {...}
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from neat_rag.providers.llm import get_langchain_llm

# ── ragas detection ───────────────────────────────────────────────────────────
try:
    from datasets import Dataset  # type: ignore

    # ragas 0.2+ uses class-based metrics
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate  # type: ignore
        from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness  # type: ignore

        _RAGAS_NEW = True
        _RAGAS_OK = True
    except ImportError:
        # Fall back to ragas 0.1.x (legacy)
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            faithfulness,
        )

        _RAGAS_NEW = False
        _RAGAS_OK = True
except ImportError:
    _RAGAS_OK = False
    _RAGAS_NEW = False


# ── API helpers ───────────────────────────────────────────────────────────────

def _create_session(api_url: str, api_key: str = "") -> str:
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = httpx.post(
        f"{api_url}/sessions",
        json={"title": "eval-session"},
        headers=headers,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def _chat(
    api_url: str,
    session_id: str,
    question: str,
    search_type: str = "hybrid",
    api_key: str = "",
) -> Dict[str, Any]:
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = httpx.post(
        f"{api_url}/chat",
        json={"message": question, "session_id": session_id, "search_type": search_type},
        headers=headers,
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()


def _list_documents(api_url: str, api_key: str = "") -> List[Dict]:
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = httpx.get(f"{api_url}/documents?limit=10", headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json().get("items", [])


def _delete_session(api_url: str, session_id: str, api_key: str = "") -> None:
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        httpx.delete(f"{api_url}/sessions/{session_id}", headers=headers, timeout=5)
    except Exception:
        pass


# ── Dataset builder ───────────────────────────────────────────────────────────

def _build_rows(
    questions: List[str],
    ground_truths: List[str],
    api_url: str,
    api_key: str,
    search_type: str,
    verbose: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths), 1):
        if verbose:
            print(f"  [{i}/{len(questions)}] {question[:70]}…")

        try:
            session_id = _create_session(api_url, api_key)
            result = _chat(api_url, session_id, question, search_type, api_key)
            _delete_session(api_url, session_id, api_key)
        except Exception as exc:
            print(f"    ✗ API error: {exc}")
            continue

        answer = result.get("content", "")
        citations = result.get("citations", [])
        contexts = [c["content_snippet"] for c in citations] or [""]

        if verbose:
            print(f"    answer ({len(answer)} chars), citations: {len(citations)}")

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    return rows


# ── Ragas evaluation ──────────────────────────────────────────────────────────

def _evaluate_ragas_new(rows: List[Dict], llm_model: str, verbose: bool) -> Dict:
    """ragas >= 0.2 path."""
    from ragas.llms import LangchainLLMWrapper  # type: ignore

    evaluator_llm = LangchainLLMWrapper(get_langchain_llm(model=llm_model))

    samples = [
        SingleTurnSample(  # type: ignore[name-defined]
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            response=r["answer"],
            reference=r["ground_truth"] if r["ground_truth"] else None,
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=samples)  # type: ignore[name-defined]
    metrics = [
        Faithfulness(),          # type: ignore[name-defined]
        AnswerRelevancy(),       # type: ignore[name-defined]
        ContextPrecision(),      # type: ignore[name-defined]
    ]

    if verbose:
        print("  Running ragas evaluate (new API)…")

    result = evaluate(dataset, metrics=metrics, llm=evaluator_llm)  # type: ignore
    df = result.to_pandas()
    return {
        "aggregate": df.mean(numeric_only=True).to_dict(),
        "per_sample": df.to_dict(orient="records"),
    }


def _evaluate_ragas_legacy(rows: List[Dict], verbose: bool) -> Dict:
    """ragas 0.1.x path."""
    dataset = Dataset.from_list(rows)  # type: ignore[name-defined]
    metrics = [
        faithfulness,       # type: ignore[name-defined]
        answer_relevancy,   # type: ignore[name-defined]
        context_precision,  # type: ignore[name-defined]
    ]

    if verbose:
        print("  Running ragas evaluate (legacy API)…")

    result = evaluate(dataset, metrics=metrics)  # type: ignore
    df = result.to_pandas()
    return {
        "aggregate": df.mean(numeric_only=True).to_dict(),
        "per_sample": df.to_dict(orient="records"),
    }


def _evaluate_basic(rows: List[Dict], verbose: bool) -> Dict:
    """Heuristic evaluation when ragas is not available."""
    if verbose:
        print("  ragas not installed — running heuristic evaluation.")

    per_sample = []
    for r in rows:
        answer = r["answer"]
        contexts = r["contexts"]
        question = r["question"]

        # Simple heuristics (not a substitute for ragas)
        answer_len = len(answer.split())
        has_context_overlap = any(
            any(w in answer.lower() for w in ctx.lower().split()[:20])
            for ctx in contexts
        )
        question_keywords = [w for w in question.lower().split() if len(w) > 3]
        answer_addresses = sum(k in answer.lower() for k in question_keywords) / max(
            len(question_keywords), 1
        )

        per_sample.append(
            {
                "question": question,
                "answer_word_count": answer_len,
                "has_context_overlap": has_context_overlap,
                "question_coverage": round(answer_addresses, 3),
                "num_citations": len([c for c in contexts if c]),
            }
        )

    avg_coverage = sum(s["question_coverage"] for s in per_sample) / max(len(per_sample), 1)
    pct_overlap = sum(1 for s in per_sample if s["has_context_overlap"]) / max(len(per_sample), 1)

    return {
        "aggregate": {
            "avg_question_coverage": round(avg_coverage, 3),
            "pct_context_overlap": round(pct_overlap, 3),
            "note": "Install ragas for full faithfulness/relevancy/precision scores.",
        },
        "per_sample": per_sample,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_evaluation(
    questions: List[str],
    ground_truths: Optional[List[str]] = None,
    api_url: str = "http://localhost:8058",
    api_key: str = "",
    search_type: str = "hybrid",
    eval_llm_model: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full evaluation pipeline and return a report dict.

    Args:
        questions:       Questions to evaluate.
        ground_truths:   Reference answers (optional; empty strings if omitted).
        api_url:         Neat-RAG API base URL.
        api_key:         API key for authenticated deployments.
        search_type:     "hybrid" or "vector".
        eval_llm_model:  The model used by ragas for LLM-as-judge evaluation.
                         Defaults to settings.EVAL_LLM_MODEL. 
                         To use DeepSeek/Gemini, set configuration in .env or config.py.
        verbose:         Print progress to stdout.
    """
    from neat_rag.config import settings
    eval_llm_model = eval_llm_model or settings.EVAL_LLM_MODEL

    if ground_truths is None:
        ground_truths = [""] * len(questions)

    if len(ground_truths) < len(questions):
        ground_truths += [""] * (len(questions) - len(ground_truths))

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Neat-RAG Evaluation  |  {len(questions)} question(s)")
        print(f"  API: {api_url}  |  mode: {search_type}")
        print(f"{'─'*55}")

    # Collect answers from the API
    if verbose:
        print("\nStep 1 — Querying RAG pipeline…")
    rows = _build_rows(questions, ground_truths, api_url, api_key, search_type, verbose)

    if not rows:
        print("No results collected — aborting.")
        sys.exit(1)

    if verbose:
        print(f"\nStep 2 — Evaluating {len(rows)} sample(s)…")

    # Run evaluation
    if _RAGAS_OK and _RAGAS_NEW:
        scores = _evaluate_ragas_new(rows, eval_llm_model, verbose)
    elif _RAGAS_OK:
        scores = _evaluate_ragas_legacy(rows, verbose)
    else:
        scores = _evaluate_basic(rows, verbose)

    # Assemble report
    from neat_rag.config import settings

    report: Dict[str, Any] = {
        "samples": len(rows),
        "aggregate": scores["aggregate"],
        "per_sample": scores["per_sample"],
        "config": {
            "api_url": api_url,
            "search_type": search_type,
            "llm_model": settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "enable_rerank": settings.ENABLE_RERANK,
            "reranker_model": settings.RERANKER_MODEL if settings.ENABLE_RERANK else None,
        },
    }

    if verbose:
        print(f"\n{'─'*55}")
        print("  Results")
        print(f"{'─'*55}")
        for k, v in report["aggregate"].items():
            if isinstance(v, float):
                print(f"  {k:<30} {v:.4f}")
            else:
                print(f"  {k:<30} {v}")
        print(f"{'─'*55}\n")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

_DEMO_QUESTIONS = [
    "What is the main topic covered by the documents?",
    "Summarize the key findings or conclusions.",
    "What recommendations or action items are mentioned?",
    "Are there any statistics or numerical data mentioned?",
    "Who are the main stakeholders or target audience?",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neat-RAG ragas evaluation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m neat_rag.eval --demo
  python -m neat_rag.eval --questions questions.json --output report.json
  python -m neat_rag.eval --questions qa.json --api-url http://prod:8058 --api-key sk-xxx
        """,
    )
    parser.add_argument("--questions", type=Path, help="JSON file with questions")
    parser.add_argument("--demo", action="store_true", help="Use built-in demo questions")
    parser.add_argument(
        "--output", type=Path, default=Path("eval_report.json"), help="Report output path"
    )
    parser.add_argument("--api-url", default="http://localhost:8058", help="Neat-RAG API URL")
    parser.add_argument("--api-key", default="", help="API key (if auth is enabled)")
    parser.add_argument(
        "--search-type", default="hybrid", choices=["hybrid", "vector"], help="Search mode"
    )
    parser.add_argument(
        "--eval-llm", default=None, help="Model for ragas evaluation (defaults to EVAL_LLM_MODEL in config)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    if not args.questions and not args.demo:
        parser.error("Provide --questions <file> or --demo")

    if args.demo:
        questions = _DEMO_QUESTIONS
        ground_truths: Optional[List[str]] = None
        if not args.quiet:
            print("Using built-in demo questions.")
    else:
        raw = json.loads(args.questions.read_text(encoding="utf-8"))  # type: ignore[union-attr]
        if isinstance(raw, list) and raw and isinstance(raw[0], str):
            questions = raw
            ground_truths = None
        else:
            questions = [d["question"] for d in raw]
            ground_truths = [d.get("ground_truth", "") for d in raw]

    report = run_evaluation(
        questions=questions,
        ground_truths=ground_truths,
        api_url=args.api_url,
        api_key=args.api_key,
        search_type=args.search_type,
        eval_llm_model=args.eval_llm,
        verbose=not args.quiet,
    )

    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"Report saved → {args.output}")


if __name__ == "__main__":
    main()
