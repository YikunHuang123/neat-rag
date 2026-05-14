#!/usr/bin/env python3
"""
Phase 2 Agent & Retrieval Test
================================
Tests the Orchestrator → Tools → Retrieval → LLM pipeline.

Usage:
    python test_agent.py

Prerequisites:
    1. PostgreSQL running and Phase 1 ingestion test has been run
       (so there are chunks in the database).
    2. .env file configured (with LLM_PROVIDER and API keys).
"""

import asyncio
import sys
from pathlib import Path

# ── path setup (src layout) ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from neat_rag.db.pool import pg_pool
from neat_rag.db.sessions import SessionRepository
from neat_rag.agent.orchestrator import run_query
from neat_rag.config import settings

def _section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")

async def main():
    _section(f"Phase 2 Agent Test (LLM: {settings.LLM_PROVIDER} / {settings.LLM_MODEL})")
    
    try:
        await pg_pool.connect()
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        return

    # Create a new chat session in the DB
    try:
        async with pg_pool.get_connection() as conn:
            session_repo = SessionRepository(conn)
            session = await session_repo.create_session(user_id="test_user", title="CLI Test Session")
    except Exception as e:
        print(f"Failed to create session in DB. Did you run Phase 1 migrations? Error: {e}")
        await pg_pool.disconnect()
        return
    
    print(f"  ✓ Session created: {session.id}")
    print(f"  ✓ Agent ready! Try asking a question about RAG (type 'quit' to exit).")
    print(f"{'─'*50}")

    try:
        while True:
            try:
                question = input("\n\033[94mUser:\033[0m ").strip()
            except EOFError:
                break
                
            if not question:
                continue
                
            if question.lower() in ("quit", "exit", "q"):
                break
                
            print("\033[90m[Agent is thinking...]\033[0m")
            
            try:
                # Call the orchestrator
                answer = await run_query(question=question, session_id=session.id, user_id="test_user")
                print(f"\n\033[92mAgent:\033[0m {answer}")
            except Exception as e:
                import traceback
                print(f"\n\033[91mError during query:\033[0m {e}")
                traceback.print_exc()
                
    finally:
        await pg_pool.disconnect()
        print("\nSession ended.")

if __name__ == "__main__":
    asyncio.run(main())
