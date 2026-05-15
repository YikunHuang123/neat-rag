"""
Neat-RAG Streamlit UI
Connects to the Neat-RAG backend API via HTTP.

Run:
    streamlit run src/neat_rag/ui.py
"""
import json
import time
from typing import Any, Dict, Generator, List, Optional

import httpx
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neat-RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.2rem; padding-bottom: 0; max-width: 900px;}

/* ── Brand ── */
.nr-brand {
    display: flex; align-items: center; gap: 12px;
    padding: 1.1rem 0 1.4rem; border-bottom: 1px solid rgba(255,255,255,.07);
    margin-bottom: 1.2rem;
}
.nr-logo {
    width: 40px; height: 40px; border-radius: 11px; flex-shrink: 0;
    background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem; font-weight: 900; color: #fff;
    box-shadow: 0 4px 14px rgba(124,58,237,.45);
}
.nr-wordmark { line-height: 1.2; }
.nr-name {
    font-size: 1.25rem; font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.nr-tag { font-size: 0.62rem; letter-spacing: 1.8px; text-transform: uppercase; color: #6b7280; }

/* ── Status pill ── */
.nr-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 11px; border-radius: 20px; font-size: 0.72rem; font-weight: 600;
}
.nr-pill-dot { width: 7px; height: 7px; border-radius: 50%; }
.nr-online  { background: rgba(52,211,153,.1);  color: #34D399; border: 1px solid rgba(52,211,153,.22); }
.nr-offline { background: rgba(239,68,68,.1);   color: #EF4444; border: 1px solid rgba(239,68,68,.22); }
.nr-online  .nr-pill-dot { background: #34D399; box-shadow: 0 0 6px #34D399; }
.nr-offline .nr-pill-dot { background: #EF4444; }

/* ── Section heading ── */
.nr-section {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 1.8px;
    text-transform: uppercase; color: #4b5563; margin: 1.3rem 0 0.55rem;
}

/* ── Citation card ── */
.nr-cite {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 0.65rem 1rem; margin: 0.3rem 0; border-radius: 0.55rem;
    background: rgba(124,58,237,.06); border: 1px solid rgba(124,58,237,.15);
    border-left: 3px solid #7C3AED;
}
.nr-cite-num {
    font-weight: 800; font-size: 0.78rem; color: #7C3AED;
    background: rgba(124,58,237,.14); border-radius: 4px;
    padding: 1px 6px; flex-shrink: 0; margin-top: 2px;
}
.nr-cite-body { flex: 1; }
.nr-cite-title { font-size: 0.78rem; font-weight: 600; color: #a78bfa; margin-bottom: 3px; }
.nr-cite-snippet { font-size: 0.73rem; color: #6b7280; line-height: 1.55; }

/* ── Document card ── */
.nr-doc {
    padding: 0.85rem 1.1rem; margin: 0.4rem 0; border-radius: 0.75rem;
    background: rgba(255,255,255,.025); border: 1px solid rgba(255,255,255,.07);
    transition: border-color .2s;
}
.nr-doc:hover { border-color: rgba(124,58,237,.35); }
.nr-doc-title { font-size: 0.88rem; font-weight: 600; margin-bottom: 4px; }
.nr-doc-meta  { font-size: 0.71rem; color: #6b7280; }
.nr-chunk-badge {
    display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 0.67rem; font-weight: 700;
    background: rgba(124,58,237,.12); color: #8b5cf6; border: 1px solid rgba(124,58,237,.2);
}

/* ── Job row ── */
.nr-job {
    display: flex; align-items: center; gap: 10px;
    padding: 0.55rem 0.85rem; margin: 0.3rem 0; border-radius: 0.5rem;
    background: rgba(255,255,255,.025); border: 1px solid rgba(255,255,255,.07); font-size: 0.78rem;
}
.nr-job-s { font-weight: 700; padding: 2px 8px; border-radius: 20px; font-size: 0.67rem; }
.nr-pending    { background: rgba(251,191,36,.1);  color: #FBBF24; border: 1px solid rgba(251,191,36,.2); }
.nr-processing { background: rgba(99,102,241,.1);  color: #818cf8; border: 1px solid rgba(99,102,241,.2); }
.nr-completed  { background: rgba(52,211,153,.1);  color: #34D399; border: 1px solid rgba(52,211,153,.2); }
.nr-failed     { background: rgba(239,68,68,.1);   color: #EF4444; border: 1px solid rgba(239,68,68,.2); }

/* ── Upload dropzone ── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(124,58,237,.04) !important;
    border: 2px dashed rgba(124,58,237,.28) !important;
    border-radius: 1rem !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important; font-weight: 600 !important; transition: all .18s !important;
}
/* Active session button (disabled state in sidebar) */
section[data-testid="stSidebar"] .stButton > button:disabled {
    background-color: rgba(96, 165, 250, 0.15) !important;
    color: #60A5FA !important;
    border: 1px solid rgba(96, 165, 250, 0.3) !important;
    opacity: 1 !important;
    cursor: default !important;
}

/* ── Tab highlight ── */
.stTabs [aria-selected="true"] {
    color: #a78bfa !important; border-bottom: 2px solid #7C3AED !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div { background: linear-gradient(90deg, #7C3AED, #EC4899) !important; }

/* ── Chat Input Focus ── */
[data-testid="stChatInput"] {
    border-radius: 12px !important;
}
/* Ensure the outer container and inner div use the same uniform border */
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {
    transition: border-color 0.2s ease-in-out !important;
}
[data-testid="stChatInput"]:focus-within, [data-testid="stChatInput"]:focus-within > div {
    border: 1px solid #60A5FA !important;
    box-shadow: none !important;
    outline: none !important;
}
/* Clean up the textarea internal styles */
[data-testid="stChatInput"] textarea {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session-state defaults ────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "api_url": "http://localhost:8058",
    "api_key": "",
    "search_type": "hybrid",
    "current_session_id": None,
    "sessions": [],
    "messages": [],
    "nav": "chat",
    "api_online": False,
    "_health_ts": 0.0,
    "_sessions_dirty": True,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── API helpers ───────────────────────────────────────────────────────────────

def _hdrs() -> Dict[str, str]:
    h: Dict[str, str] = {}
    if st.session_state.api_key:
        h["X-API-Key"] = st.session_state.api_key
    return h


def _api(method: str, path: str, silent: bool = False, **kwargs) -> Optional[Dict]:
    url = st.session_state.api_url.rstrip("/") + path
    try:
        with httpx.Client(timeout=20) as c:
            resp = getattr(c, method)(url, headers=_hdrs(), **kwargs)
            if resp.status_code == 204:
                return {}
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        if not silent:
            st.error(f"API error — {exc}")
        return None


def _health() -> bool:
    now = time.time()
    if now - st.session_state._health_ts < 8:
        return st.session_state.api_online
    try:
        with httpx.Client(timeout=3) as c:
            r = c.get(st.session_state.api_url.rstrip("/") + "/health/live")
            ok = r.status_code == 200
    except Exception:
        ok = False
    st.session_state.api_online = ok
    st.session_state._health_ts = now
    return ok


def _refresh_sessions():
    data = _api("get", "/sessions?limit=50", silent=True)
    st.session_state.sessions = data.get("items", []) if data else []
    st.session_state._sessions_dirty = False


def _create_session(title: str = "New Chat") -> Optional[Dict]:
    sess = _api("post", "/sessions", json={"title": title})
    if sess:
        st.session_state._sessions_dirty = True
    return sess


def _delete_session(sid: str):
    _api("delete", f"/sessions/{sid}", silent=True)
    st.session_state._sessions_dirty = True


def _load_history(sid: str):
    data = _api("get", f"/sessions/{sid}/messages", silent=True)
    items = data.get("items", []) if data else []
    st.session_state.messages = [
        {
            "role": m["role"],
            "content": m["content"],
            "citations": [],
            "message_id": m["id"],
        }
        for m in items
    ]


def _sse_generator(
    message: str,
    session_id: str,
    search_type: str,
    result: Dict,
) -> Generator[str, None, None]:
    """Yield text deltas from /chat/stream; write citations+message_id into `result`."""
    url = st.session_state.api_url.rstrip("/") + "/chat/stream"
    payload = {"message": message, "session_id": session_id, "search_type": search_type}
    try:
        with httpx.Client(timeout=120) as c:
            with c.stream("POST", url, json=payload, headers=_hdrs()) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        return
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    t = ev.get("type")
                    if t == "delta":
                        yield ev.get("content", "")
                    elif t == "done":
                        result["message_id"] = ev.get("message_id")
                        result["citations"] = ev.get("citations", [])
                        result["title"] = ev.get("title")
                    elif t == "error":
                        raise RuntimeError(ev.get("content", "Stream error"))
    except Exception as exc:
        yield f"\n\n_(Stream error: {exc})_"


def _render_citations(citations: List[Dict]):
    if not citations:
        return
    with st.expander(f"Sources  ·  {len(citations)} citation(s)", expanded=False):
        for c in citations:
            st.markdown(f"""
<div class="nr-cite">
  <span class="nr-cite-num">[{c['citation_number']}]</span>
  <div class="nr-cite-body">
    <div class="nr-cite-title">{c['document_title']}</div>
    <div class="nr-cite-snippet">{c['content_snippet']}</div>
  </div>
</div>""", unsafe_allow_html=True)


def _render_feedback(message_id: Optional[str], key_suffix: str):
    if not message_id:
        return
    c1, c2, _ = st.columns([1, 1, 10])
    with c1:
        if st.button("👍", key=f"up_{key_suffix}", help="Good answer"):
            _api("post", "/feedback", json={"message_id": message_id, "rating": 1}, silent=True)
            st.toast("Thanks for the feedback!")
    with c2:
        if st.button("👎", key=f"dn_{key_suffix}", help="Poor answer"):
            _api("post", "/feedback", json={"message_id": message_id, "rating": -1}, silent=True)
            st.toast("Feedback noted.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    online = _health()
    pill_cls = "nr-online" if online else "nr-offline"
    pill_txt = "Connected" if online else "Offline"

    st.markdown(f"""
<div class="nr-brand">
  <div class="nr-logo">N</div>
  <div class="nr-wordmark">
    <div class="nr-name">Neat-RAG</div>
    <div class="nr-tag">Agentic Retrieval</div>
  </div>
</div>
<div style="margin-bottom:.9rem">
  <span class="nr-pill {pill_cls}">
    <span class="nr-pill-dot"></span>{pill_txt}
  </span>
</div>""", unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="nr-section">Navigate</div>', unsafe_allow_html=True)
    nav_cols = st.columns(2)
    with nav_cols[0]:
        if st.button(
            "💬  Chat",
            use_container_width=True,
            type="primary" if st.session_state.nav == "chat" else "secondary",
        ):
            st.session_state.nav = "chat"
            st.rerun()
    with nav_cols[1]:
        if st.button(
            "📚  Docs",
            use_container_width=True,
            type="primary" if st.session_state.nav == "documents" else "secondary",
        ):
            st.session_state.nav = "documents"
            st.rerun()

    # Session list (chat page only)
    if st.session_state.nav == "chat":
        st.markdown('<div class="nr-section">Sessions</div>', unsafe_allow_html=True)

        if st.button("＋  New session", use_container_width=True):
            sess = _create_session()
            if sess:
                st.session_state.current_session_id = sess["id"]
                st.session_state.messages = []
                _refresh_sessions()
                st.rerun()

        if st.session_state._sessions_dirty:
            _refresh_sessions()

        for s in st.session_state.sessions:
            is_active = s["id"] == st.session_state.current_session_id
            sc, dc = st.columns([5, 1])
            with sc:
                raw = s["title"]
                short = raw if len(raw) <= 18 else raw[:17] + "…"
                label = f"▸ {short}" if is_active else short
                if st.button(label, key=f"s_{s['id']}", use_container_width=True, disabled=is_active):
                    st.session_state.current_session_id = s["id"]
                    _load_history(s["id"])
                    st.rerun()
            with dc:
                if st.button("✕", key=f"ds_{s['id']}", help="Delete"):
                    _delete_session(s["id"])
                    if st.session_state.current_session_id == s["id"]:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    _refresh_sessions()
                    st.rerun()

    # Settings
    st.markdown('<div class="nr-section">Settings</div>', unsafe_allow_html=True)

    new_url = st.text_input(
        "API URL", value=st.session_state.api_url,
        placeholder="http://localhost:8058", label_visibility="collapsed",
    )
    if new_url != st.session_state.api_url:
        st.session_state.api_url = new_url
        st.session_state._health_ts = 0.0
        st.rerun()

    new_key = st.text_input(
        "API Key", value=st.session_state.api_key,
        placeholder="API key (optional)", type="password", label_visibility="collapsed",
    )
    if new_key != st.session_state.api_key:
        st.session_state.api_key = new_key

    st.session_state.search_type = st.selectbox(
        "Search mode",
        options=["hybrid", "vector"],
        index=0 if st.session_state.search_type == "hybrid" else 1,
    )

    if st.button("⟳  Refresh status", use_container_width=True):
        st.session_state._health_ts = 0.0
        st.session_state._sessions_dirty = True
        st.rerun()


# ── Chat page ─────────────────────────────────────────────────────────────────
if st.session_state.nav == "chat":
    sid = st.session_state.current_session_id

    if sid is None:
        st.markdown("## Chat")
        st.info("Create or select a session in the sidebar to start chatting.")
    else:
        # Resolve title
        title = next(
            (s["title"] for s in st.session_state.sessions if s["id"] == sid),
            "Chat",
        )
        st.markdown(f"## {title}")
        st.caption(f"Session `{sid[:8]}…`  ·  mode: **{st.session_state.search_type}**")
        st.divider()

        # Render history
        for i, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            avatar = "🧑" if role == "user" else "🤖"
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])
                if role != "user":
                    _render_citations(msg.get("citations", []))
                    _render_feedback(msg.get("message_id"), key_suffix=f"hist_{i}")

        # Input
        if prompt := st.chat_input("Ask something about your documents…"):
            st.session_state.messages.append(
                {"role": "user", "content": prompt, "citations": [], "message_id": None}
            )
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🤖"):
                result: Dict = {"message_id": None, "citations": []}
                answer = st.write_stream(
                    _sse_generator(prompt, sid, st.session_state.search_type, result)
                )
                _render_citations(result["citations"])
                _render_feedback(result["message_id"], key_suffix=f"new_{len(st.session_state.messages)}")

            st.session_state.messages.append({
                "role": "agent",
                "content": answer,
                "citations": result["citations"],
                "message_id": result["message_id"],
            })

            if result.get("title"):
                for s in st.session_state.sessions:
                    if s["id"] == sid:
                        s["title"] = result["title"]
                        break
                st.rerun()


# ── Documents page ────────────────────────────────────────────────────────────
elif st.session_state.nav == "documents":
    st.markdown("## Document Library")

    _MIME_ICON = {
        "application/pdf": "📄",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "📝",
        "text/markdown": "📋",
        "text/plain": "📃",
        "text/html": "🌐",
    }

    tab_lib, tab_upload, tab_jobs = st.tabs(["Library", "Upload", "Jobs"])

    # ── Library tab ───────────────────────────────────────────────────────────
    with tab_lib:
        col_hdr, col_ref = st.columns([5, 1])
        with col_ref:
            if st.button("⟳ Refresh", use_container_width=True):
                st.rerun()

        data = _api("get", "/documents?limit=100")
        if data:
            docs = data.get("items", [])
            total = data.get("total", len(docs))
            with col_hdr:
                st.markdown(f"**{total}** document(s) in your knowledge base")

            if not docs:
                st.info("No documents yet — use the Upload tab to add some.")
            else:
                for doc in docs:
                    icon = _MIME_ICON.get(doc.get("mime_type", ""), "📁")
                    created = doc.get("created_at", "")[:10]
                    dcol, bcol = st.columns([5, 1])
                    with dcol:
                        st.markdown(f"""
<div class="nr-doc">
  <div class="nr-doc-title">{icon}&nbsp; {doc['title']}</div>
  <div class="nr-doc-meta">
    {doc.get('source', '')} &nbsp;·&nbsp; {created}
    &nbsp;·&nbsp; <span class="nr-chunk-badge">{doc.get('chunk_count', 0)} chunks</span>
  </div>
</div>""", unsafe_allow_html=True)
                    with bcol:
                        if st.button("🗑", key=f"del_{doc['id']}", help="Delete document"):
                            res = _api("delete", f"/documents/{doc['id']}")
                            if res is not None:
                                st.success(f"Deleted '{doc['title']}'")
                                st.rerun()

    # ── Upload tab ────────────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("Supported formats: **PDF · DOCX · Markdown · TXT · HTML** &nbsp;(max 50 MB each)")

        files = st.file_uploader(
            "Drop files here or click to browse",
            type=["pdf", "docx", "md", "txt", "html"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if files:
            st.markdown(f"**{len(files)} file(s) selected:**")
            for f in files:
                st.caption(f"· {f.name}  ({f.size / 1024:.0f} KB)")

            if st.button("Ingest selected files", type="primary"):
                for f in files:
                    with st.spinner(f"Uploading {f.name}…"):
                        try:
                            mime = f.type or "application/octet-stream"
                            with httpx.Client(timeout=30) as c:
                                resp = c.post(
                                    st.session_state.api_url.rstrip("/") + "/documents/upload",
                                    files={"file": (f.name, f.getvalue(), mime)},
                                    headers=_hdrs(),
                                )
                                resp.raise_for_status()
                                jdata = resp.json()
                            st.success(f"✓ **{f.name}** → job `{jdata['job_id']}`")
                        except Exception as exc:
                            st.error(f"✗ {f.name}: {exc}")

    # ── Jobs tab ──────────────────────────────────────────────────────────────
    with tab_jobs:
        jcol, rjcol = st.columns([5, 1])
        with rjcol:
            if st.button("⟳ Refresh", key="ref_jobs", use_container_width=True):
                st.rerun()

        jdata = _api("get", "/jobs?limit=30")
        if jdata:
            jobs = jdata.get("items", [])
            with jcol:
                st.markdown(f"**{len(jobs)}** recent job(s)")

            if not jobs:
                st.info("No ingestion jobs yet.")
            else:
                for job in jobs:
                    status = job.get("status", "pending")
                    progress = job.get("progress", 0.0)
                    created = job.get("created_at", "")[:10]

                    jrow, jprog = st.columns([3, 2])
                    with jrow:
                        st.markdown(f"""
<div class="nr-job">
  <span class="nr-job-s nr-{status}">{status.upper()}</span>
  <span style="flex:1;color:#9ca3af">{job.get('filename','')}</span>
  <span style="color:#4b5563;font-size:.7rem">{created}</span>
</div>""", unsafe_allow_html=True)
                    with jprog:
                        if status in ("processing", "completed"):
                            st.progress(min(progress, 1.0))
                        if job.get("error"):
                            st.caption(f":red[{job['error']}]")
