"""
Neat-RAG Streamlit UI
Connects to the Neat-RAG backend API via HTTP.

Run:
    streamlit run src/neat_rag/ui.py
"""
import html as _html
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

import httpx
import streamlit as st
from neat_rag.config import settings

# ── Cookie Persistence Helpers ────────────────────────────────────────────────
def _get_cookie(name: str, default: Any = None) -> Any:
    """Read a cookie via st.context.cookies (Streamlit 1.34+)."""
    try:
        # st.context.cookies is a read-only dict
        return st.context.cookies.get(name, default)
    except Exception:
        return default


def _set_cookie_js(name: str, value: str):
    """Queue a cookie to be set via JS bridge at the end of the run."""
    if "_pending_cookies" not in st.session_state:
        st.session_state["_pending_cookies"] = {}
    st.session_state["_pending_cookies"][name] = value


def _render_pending_cookies():
    """Renders all queued cookies in a single JS call to avoid layout gaps."""
    pending = st.session_state.get("_pending_cookies", {})
    if not pending:
        return
    
    import json
    scripts = []
    for name, value in pending.items():
        safe_val = json.dumps(value)
        scripts.append(f'document.cookie = "{name}=" + encodeURIComponent({safe_val}) + "; path=/; max-age=31536000; SameSite=Lax";')
    
    # Use window.parent.document because Streamlit components run in an iframe
    js_code = f"""
        <script>
            const setCookie = (n, v) => {{
                window.parent.document.cookie = n + "=" + encodeURIComponent(v) + "; path=/; max-age=31536000; SameSite=Lax";
            }};
            {chr(10).join([f"setCookie('{k}', {json.dumps(v)});" for k, v in pending.items()])}
        </script>
    """
    st.iframe(js_code, height=1)
    # Clear queue after rendering
    st.session_state["_pending_cookies"] = {}

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
#MainMenu, footer {visibility: hidden;}
/* Keep header visible to ensure sidebar toggle works, but hide its background if needed */
header { background-color: transparent !important; }
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

/* Fix for centering icons (Emoji) in small buttons, especially in sidebar */
[data-testid="stSidebar"] .stButton button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    height: 30px !important;
    min-width: 30px !important; /* Smaller square */
    width: 100% !important;
    font-size: 0.85rem !important; /* Smaller icon font */
}
[data-testid="stSidebar"] .stButton button p {
    margin: 0 !important;
    line-height: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
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

/* ── Inline citation refs ── */
.nr-cref {
    display: inline-block; text-decoration: none;
    color: #a78bfa; font-size: 0.72em; font-weight: 700;
    vertical-align: super; cursor: pointer;
    border: 1px solid rgba(167,139,250,.35); border-radius: 3px;
    padding: 0 3px; margin: 0 1px; line-height: 1;
    transition: background .15s;
}
.nr-cref:hover { background: rgba(124,58,237,.2); }

/* ── Citation modal overlay ── */
.nr-covl {
    display: none;
    position: fixed; top: 0; right: 0; bottom: 0; left: 0;
    background: rgba(0,0,0,.58);
    z-index: 99999;
    justify-content: center; align-items: center;
}
.nr-covl:target { display: flex; }
.nr-covl-bg { position: absolute; top: 0; right: 0; bottom: 0; left: 0; }

/* ── Citation modal box ── */
.nr-cbox {
    position: relative; z-index: 1;
    background: #141428; border-radius: 14px;
    padding: 1.4rem 1.6rem; max-width: 620px; width: 90%;
    border: 1px solid rgba(124,58,237,.4);
    box-shadow: 0 24px 64px rgba(0,0,0,.7);
}
.nr-cbox-hdr {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: .85rem;
}
.nr-cbox-num { color: #a78bfa; font-weight: 800; font-size: .95rem; }
.nr-cbox-x {
    color: #6b7280; text-decoration: none; font-size: 1rem;
    padding: 3px 7px; border-radius: 5px; transition: all .15s;
}
.nr-cbox-x:hover { color: #e5e7eb; background: rgba(255,255,255,.09); }
.nr-cbox-title { color: #e2e8f0; font-weight: 700; font-size: .9rem; margin-bottom: .25rem; }
.nr-cbox-src { color: #6b7280; font-size: .71rem; margin-bottom: .95rem; }
.nr-cbox-body {
    color: #d1d5db; font-size: .83rem; line-height: 1.65;
    background: rgba(255,255,255,.04); padding: .85rem 1rem;
    border-radius: 8px; border-left: 3px solid #7C3AED;
    white-space: pre-wrap; word-break: break-word;
    max-height: 600px; overflow-y: auto;
}

/* ── Center modal in chat area when sidebar is open ── */
/* When stSidebarCollapsedControl is absent → sidebar is open → shift box right by half sidebar width */
html:not(:has([data-testid="stSidebarCollapsedControl"])) .nr-cbox {
    transform: translateX(10.5rem);
}

/* ── Keep sidebar toggle buttons visible ── */
[data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebar"] button {
    visibility: visible !important;
}

/* ── Chat message hover highlight ── */
[data-testid="stChatMessage"] {
    border-radius: 10px !important;
    transition: background .18s;
    margin-right: -20px !important;
    padding-right: 20px !important;
}
[data-testid="stChatMessage"]:hover {
    background: rgba(0, 0, 0, 0.1) !important;
}

/* ── User message default background ── */
[data-testid="stChatMessage"]:has([aria-label*="user" i]) {
    background-color: rgba(0, 0, 0, 0.08) !important;
}

/* ── Feedback row: hidden by default, revealed on message hover ── */
[data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] {
    opacity: 0;
    pointer-events: none;
    transition: opacity .15s ease;
    max-width: 85px;
    margin-top: 4px !important;
    gap: 5px !important;
}
[data-testid="stChatMessage"]:hover [data-testid="stHorizontalBlock"] {
    opacity: 1;
    pointer-events: auto;
}


/* ── Feedback buttons: small pill style ── */
[data-testid="stChatMessage"] [data-testid="baseButton-secondary"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    font-size: 0.8rem !important;
    min-height: 22px !important;
    height: 22px !important;
    line-height: 1 !important;
    border-radius: 11px !important;
    background: transparent !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    color: #9ca3af !important;
    width: 100% !important;
}
[data-testid="stChatMessage"] [data-testid="baseButton-secondary"] p {
    margin: 0 !important;
    line-height: 1 !important;
}
[data-testid="stChatMessage"] [data-testid="baseButton-secondary"]:hover {
    background: rgba(255,255,255,.09) !important;
    color: #e5e7eb !important;
    border-color: rgba(255,255,255,.28) !important;
}
</style>
""", unsafe_allow_html=True)


# ── LLM provider / model catalogue ───────────────────────────────────────────
_PROVIDER_OPTS: Dict[str, str] = {
    "server":    "Server default",
    "openai":    "OpenAI",
    "gemini":    "Google Gemini",
    "anthropic": "Anthropic",
    "deepseek":  "DeepSeek",
    "ollama":    "Ollama (local)",
    "custom":    "Custom endpoint",
}
_MODEL_DEFAULTS: Dict[str, str] = {
    "openai":    "gpt-4o-mini",
    "gemini":    "gemini-2.5-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "deepseek":  "deepseek-chat",
    "ollama":    "qwen2.5:7b",
    "custom":    "",
}

# ── Session-state defaults ────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "api_url": _get_cookie("nrag_api_url", os.getenv("API_URL", "http://localhost:8058")),
    "api_key": _get_cookie("nrag_api_key", ""),
    "search_type": settings.DEFAULT_SEARCH_TYPE,
    "current_session_id": None,
    "sessions": [],
    "messages": [],
    "nav": "chat",
    "api_online": False,
    "_health_ts": 0.0,
    "_sessions_dirty": True,
    "current_user": None,
    "api_key_valid": None,   # None=unknown, True=valid, False=invalid
    "is_admin": False,       # True when current key has admin access
    "_redeemed_key": None,
    "_invite_token_result": None,
    "session_page": 1,
    "session_page_size": 10,
    "session_total": 0,
    # LLM model override (set from the Model picker in the sidebar)
    "llm_provider": _get_cookie("nrag_llm_provider", "server"),
    "llm_model_inp": _get_cookie("nrag_llm_model", ""),
    "llm_api_key_inp": _get_cookie("nrag_llm_api_key", ""),
    "llm_base_url_inp": _get_cookie("nrag_llm_base_url", ""),
    "editing_session_id": None,
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
            if resp.status_code == 401:
                if not silent:
                    st.warning("Authentication required — please enter your API key in the Account section.")
                return None
            if resp.status_code == 403:
                if not silent:
                    try:
                        detail = resp.json().get("detail", "Access denied")
                    except Exception:
                        detail = "Access denied"
                    st.error(f"Access denied — {detail}. Please contact your administrator.")
                return None
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        if not silent:
            err_msg = str(exc)
            if "header value" in err_msg.lower():
                st.error("API Key error — the provided key contains invalid characters or spaces.")
            else:
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



def _require_auth():
    """Gate: stop rendering if user has no key or an invalid key."""
    if not st.session_state.api_key:
        st.info("Enter your API key in the **Account** section of the sidebar to get started.")
        st.stop()
    if st.session_state.api_key_valid is False and st.session_state.api_online:
        st.error("Your API key is invalid — please check the **Account** section.")
        st.stop()


def _refresh_sessions():
    size = st.session_state.session_page_size
    offset = (st.session_state.session_page - 1) * size
    data = _api("get", f"/sessions?limit={size}&offset={offset}", silent=True)
    if data is not None:
        st.session_state.api_key_valid = True
        st.session_state.sessions = data.get("items", [])
        st.session_state.session_total = data.get("total", 0)
    else:
        if st.session_state.api_online:
            st.session_state.api_key_valid = False
        st.session_state.sessions = []
        st.session_state.session_total = 0

    if st.session_state.sessions and not st.session_state.current_user:
        st.session_state.current_user = st.session_state.sessions[0].get("user_id")
    st.session_state._sessions_dirty = False


def _create_session(title: str = "New Chat") -> Optional[Dict]:
    sess = _api("post", "/sessions", json={"title": title})
    if sess:
        st.session_state._sessions_dirty = True
        st.session_state.session_page = 1  # Reset to page 1 to see new session
        if not st.session_state.current_user and sess.get("user_id"):
            st.session_state.current_user = sess["user_id"]
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
    payload: Dict[str, Any] = {"message": message, "session_id": session_id, "search_type": search_type}
    # Attach LLM override when the user has chosen a specific provider
    if st.session_state.llm_provider != "server":
        payload["llm_provider"] = st.session_state.llm_provider
        _m = st.session_state.get("llm_model_inp") or _MODEL_DEFAULTS.get(st.session_state.llm_provider, "")
        if _m:
            payload["llm_model"] = _m
        if st.session_state.get("llm_api_key_inp"):
            payload["llm_api_key"] = st.session_state.llm_api_key_inp
        if st.session_state.get("llm_base_url_inp"):
            payload["llm_base_url"] = st.session_state.llm_base_url_inp
    try:
        with httpx.Client(timeout=120) as c:
            with c.stream("POST", url, json=payload, headers=_hdrs()) as resp:
                if resp.status_code == 403:
                    try:
                        detail = json.loads(resp.read()).get("detail", "Access denied")
                    except Exception:
                        detail = "Access denied"
                    raise PermissionError(f"Access denied — {detail}. Please contact your administrator.")
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


def _render_message_with_citations(content: str, citations: List[Dict], msg_key: str) -> None:
    """Render message text with [n] replaced by clickable links that open a CSS :target modal."""
    if not citations:
        st.markdown(content)
        return

    cite_map = {c["citation_number"]: c for c in citations}

    def _replace(m: re.Match) -> str:
        n = int(m.group(1))
        if n not in cite_map:
            return m.group(0)
        return f'<a href="#nr-cm-{msg_key}-{n}" class="nr-cref">[{n}]</a>'

    text_html = re.sub(r'\[(\d+)\]', _replace, content)

    # Close anchor
    close_id = f"nr-cc-{msg_key}"
    close_anchor = f'<span id="{close_id}"></span>'

    modals = ""
    for n in sorted(cite_map):
        c = cite_map[n]
        modals += (
            f'<div id="nr-cm-{msg_key}-{n}" class="nr-covl">'
            f'<a href="#{close_id}" class="nr-covl-bg"></a>'
            f'<div class="nr-cbox">'
            f'<div class="nr-cbox-hdr">'
            f'<span class="nr-cbox-num">[{n}]</span>'
            f'<a href="#{close_id}" class="nr-cbox-x">✕</a>'
            f'</div>'
            f'<div class="nr-cbox-title">{_html.escape(c["document_title"])}</div>'
            f'<div class="nr-cbox-src">{_html.escape(c["document_source"])}</div>'
            f'<div class="nr-cbox-body">{_html.escape(c["content_snippet"])}</div>'
            f'</div>'
            f'</div>'
        )

    st.markdown(close_anchor + text_html + modals, unsafe_allow_html=True)


def _render_feedback(message_id: Optional[str], key_suffix: str):
    if not message_id:
        return
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("👍", key=f"up_{key_suffix}", help="Good answer"):
            _api("post", "/feedback", json={"message_id": message_id, "rating": 1}, silent=True)
            st.toast("Thanks for the feedback!")
    with c2:
        if st.button("👎", key=f"dn_{key_suffix}", help="Poor answer"):
            _api("post", "/feedback", json={"message_id": message_id, "rating": -1}, silent=True)
            st.toast("Feedback noted.")


def _on_rename_submit(sid: str):
    """Callback for renaming a session via text_input (on_change)."""
    val_key = f"edit_val_{sid}"
    if val_key in st.session_state:
        new_title = st.session_state[val_key].strip()
        if new_title:
            _api("patch", f"/sessions/{sid}", json={"title": new_title})
            _refresh_sessions()
    st.session_state.editing_session_id = None


def _is_likely_downloading(job: Dict) -> bool:
    """True when a job is likely stalled on a first-run model download (Docling / SmolVLM)."""
    if job.get("status") not in ("pending", "processing"):
        return False
    if job.get("progress", 0.0) > 0.15:
        return False
    created = job.get("created_at", "")
    if not created:
        return False
    try:
        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() > 20
    except Exception:
        return False


# ── Tasks panel (fragment — auto-refreshes every 3 s while page is open) ───────
@st.fragment(run_every=3)
def _jobs_panel():
    jcol, rjcol, cjcol = st.columns([4, 1, 1])
    with rjcol:
        if st.button("⟳ Refresh", key="ref_jobs", use_container_width=True):
            st.rerun(scope="fragment")
    with cjcol:
        if st.button("🗑 Clear", key="clr_jobs", use_container_width=True, help="Clear all background tasks"):
            _api("delete", "/jobs", silent=True)
            st.rerun(scope="fragment")

    jdata = _api("get", "/jobs?limit=30", silent=True)
    jobs = (jdata or {}).get("items", [])
    has_active = any(j.get("status") in ("pending", "processing") for j in jobs)

    with jcol:
        live_badge = (
            ' &nbsp;<span style="color:#34D399;font-size:0.7rem;font-weight:700">● Live</span>'
            if has_active else ""
        )
        st.markdown(f"**{len(jobs)}** recent task(s){live_badge}", unsafe_allow_html=True)

    if not jobs:
        st.info("No background tasks yet.")
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
                if status == "pending":
                    st.progress(0.0, text="Queued…")
                elif status == "processing":
                    pct = min(progress, 1.0)
                    st.progress(pct, text=f"{pct * 100:.0f}%")
                elif status == "completed":
                    st.progress(1.0, text="✓ Done")
                if job.get("error"):
                    st.caption(f":red[{job['error']}]")
                if _is_likely_downloading(job):
                    st.caption("First run with images — downloading ~1 GB model weights, please wait…")


def _verify_api_key_logic(online: bool):
    """Probes the API to verify the current key and its permissions."""
    if st.session_state.api_key and st.session_state.api_key_valid is None:
        # 1. Try regular access first to verify key validity
        res = _api("get", "/sessions?limit=1", silent=True)
        if res is not None:
            st.session_state.api_key_valid = True
            if res.get("items"):
                st.session_state.current_user = res["items"][0].get("user_id")

            # 2. Then probe for admin privileges
            admin_res = _api("get", "/admin/keys", silent=True)
            st.session_state.is_admin = admin_res is not None
        else:
            # If API is online but request failed, the key is invalid
            if online:
                st.session_state.api_key_valid = False
    elif not st.session_state.api_key:
        st.session_state.is_admin = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    online = _health()
    pill_cls = "nr-online" if online else "nr-offline"
    pill_txt = "Connected" if online else "Offline"

    # Header with Brand and Admin button
    hcol1, hcol2 = st.columns([4, 1], vertical_alignment="center")
    with hcol1:
        st.markdown(f"""
<div class="nr-brand" style="border-bottom:none; margin-bottom:0; padding-bottom:0;">
  <div class="nr-logo">N</div>
  <div class="nr-wordmark">
    <div class="nr-name">Neat-RAG</div>
    <div class="nr-tag">Agentic Retrieval</div>
  </div>
</div>""", unsafe_allow_html=True)
    with hcol2:
        # Probe access once per key (cached in is_admin until key changes)
        _verify_api_key_logic(online)
        show_admin = st.session_state.is_admin
        if show_admin:
            if st.button(
                "🔑",
                help="Admin Panel",
                type="primary" if st.session_state.nav == "admin" else "secondary",
                use_container_width=True,
            ):
                st.session_state.nav = "admin"
                st.rerun()

    st.markdown(f"""
<div style="margin-bottom:.9rem; padding-top:1.1rem; border-top:1px solid rgba(255,255,255,.07);">
  <span class="nr-pill {pill_cls}">
    <span class="nr-pill-dot"></span>{pill_txt}
  </span>
</div>""", unsafe_allow_html=True)

    # ── Account ───────────────────────────────────────────────────────────────
    st.markdown('<div class="nr-section">Account</div>', unsafe_allow_html=True)

    new_key = st.text_input(
        "API Key", value=st.session_state.api_key,
        placeholder="Paste your API key here", type="password", label_visibility="collapsed",
    )
    if new_key != st.session_state.api_key:
        st.session_state.api_key = new_key.strip()
        _set_cookie_js("nrag_api_key", st.session_state.api_key)
        st.session_state.api_key_valid = None
        st.session_state.is_admin = False
        st.session_state.sessions = []
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.session_state.current_user = None
        st.session_state.session_page = 1
        st.session_state._sessions_dirty = True
        st.session_state._redeemed_key = None
        if st.session_state.nav == "admin" and not new_key:
            st.session_state.nav = "chat"
            
        # Manually trigger verification for the new key in the current run
        _verify_api_key_logic(online)
        show_admin = st.session_state.is_admin
        # Removed st.rerun() to allow JS bridge to execute

    if st.session_state.api_key:
        if show_admin:
            st.caption("👑 Admin")
        elif st.session_state.api_key_valid:
            st.caption(f"👤 {st.session_state.current_user or 'Verified User'}")
        elif st.session_state.api_key_valid is False:
            st.caption("🔴 Invalid key")
    else:
        st.caption("Have an invite code? Exchange it for an API key.")
        with st.expander("🎟  Redeem invite code", expanded=True):
            invite_code = st.text_input(
                "Invite code", placeholder="Paste your invite code here",
                label_visibility="collapsed", key="_invite_input",
            )
            if st.button("Get API Key", use_container_width=True, type="primary"):
                if not invite_code.strip():
                    st.warning("Please enter an invite code.")
                else:
                    try:
                        with httpx.Client(timeout=10) as c:
                            resp = c.post(
                                st.session_state.api_url.rstrip("/") + "/auth/redeem",
                                json={"token": invite_code.strip()},
                            )
                        if resp.status_code == 201:
                            data = resp.json()
                            st.session_state._redeemed_key = data["raw_key"]
                            st.rerun()
                        elif resp.status_code == 404:
                            st.error("Invite code not found.")
                        elif resp.status_code == 410:
                            st.error(resp.json().get("detail", "Invite code is no longer valid."))
                        else:
                            st.error(f"Error {resp.status_code}: {resp.text}")
                    except Exception as exc:
                        st.error(f"Could not reach the API — {exc}")

    # ── Show newly redeemed key ───────────────────────────────────────────────
    if st.session_state._redeemed_key:
        st.success("Key issued! Copy it now — it won't be shown again.")
        st.code(st.session_state._redeemed_key, language=None)
        if st.button("✓ Saved it", use_container_width=True):
            st.session_state._redeemed_key = None
            st.rerun()

    # ── Navigation ────────────────────────────────────────────────────────────
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

    # ── Sessions (chat page only) ─────────────────────────────────────────────
    if st.session_state.nav == "chat":
        st.markdown('<div class="nr-section">Sessions</div>', unsafe_allow_html=True)

        if not st.session_state.api_key:
            # No key: wipe any stale cached sessions immediately
            st.session_state.sessions = []
            st.session_state.session_total = 0
            st.session_state.current_user = None
        else:
            if st.button("＋  New session", use_container_width=True):
                sess = _create_session()
                if sess:
                    st.session_state.current_session_id = sess["id"]
                    st.session_state.messages = []
                    _refresh_sessions()
                    st.rerun()

            # Refresh if dirty OR if we haven't verified this key yet (first render)
            if st.session_state._sessions_dirty or st.session_state.api_key_valid is None:
                _refresh_sessions()

        for s in st.session_state.sessions:
            is_active = (s["id"] == st.session_state.current_session_id)
            is_editing = (s["id"] == st.session_state.editing_session_id)

            if is_editing:
                ec, bc1, bc2 = st.columns([10, 1, 1])
                with ec:
                    st.text_input(
                        "Rename", value=s["title"], key=f"edit_val_{s['id']}",
                        label_visibility="collapsed",
                        on_change=_on_rename_submit,
                        args=(s["id"],)
                    )
                with bc1:
                    if st.button("💾", key=f"save_{s['id']}", help="Save"):
                        _on_rename_submit(s["id"])
                        st.rerun()
                with bc2:
                    if st.button("🚫", key=f"can_{s['id']}", help="Cancel"):
                        st.session_state.editing_session_id = None
                        st.rerun()
            else:
                sc, rc, dc = st.columns([10, 1, 1])
                with sc:
                    raw_t = s["title"]
                    # Sidebar truncation: 20 chars
                    short_t = raw_t if len(raw_t) <= 20 else raw_t[:19] + "…"
                    label = f"▸ {short_t}" if is_active else short_t
                    
                    # Wrap title button to allow both selection and renaming
                    # (In Streamlit, we can't easily detect double-click, 
                    # so we keep the selection as is but let the pencil be the main trigger,
                    # or allow selection first, then if already active, clicking again renames)
                    if st.button(label, key=f"s_{s['id']}", use_container_width=True, disabled=is_editing):
                        if is_active:
                            # If already selected, clicking again triggers rename
                            st.session_state.editing_session_id = s["id"]
                            st.rerun()
                        else:
                            st.session_state.current_session_id = s["id"]
                            _load_history(s["id"])
                            st.rerun()
                with rc:
                    if st.button("✏️", key=f"ren_{s['id']}", help="Rename"):
                        st.session_state.editing_session_id = s["id"]
                        st.rerun()
                with dc:
                    if st.button("✕", key=f"ds_{s['id']}", help="Delete"):
                        _delete_session(s["id"])
                        if st.session_state.current_session_id == s["id"]:
                            st.session_state.current_session_id = None
                            st.session_state.messages = []
                        
                        # If we deleted the last item on the page, go back one page
                        if len(st.session_state.sessions) <= 1 and st.session_state.session_page > 1:
                            st.session_state.session_page -= 1
                        
                        st.session_state._sessions_dirty = True
                        st.rerun()

        # ── Pagination Controls ──
        total_p = (st.session_state.session_total + st.session_state.session_page_size - 1) // st.session_state.session_page_size
        if total_p > 1:
            st.markdown('<div style="margin-top: 0.8rem;"></div>', unsafe_allow_html=True)
            pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
            with pcol1:
                if st.button("◀", key="prev_page", disabled=st.session_state.session_page <= 1, use_container_width=True):
                    st.session_state.session_page -= 1
                    st.session_state._sessions_dirty = True
                    st.rerun()
            with pcol2:
                st.markdown(
                    f"<div style='text-align:center; font-size:0.75rem; color:#6b7280; line-height:30px;'>"
                    f"{st.session_state.session_page} / {total_p}</div>",
                    unsafe_allow_html=True
                )
            with pcol3:
                if st.button("▶", key="next_page", disabled=st.session_state.session_page >= total_p, use_container_width=True):
                    st.session_state.session_page += 1
                    st.session_state._sessions_dirty = True
                    st.rerun()

    # ── Model ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="nr-section">Model</div>', unsafe_allow_html=True)

    _pkeys   = list(_PROVIDER_OPTS.keys())
    _plabels = list(_PROVIDER_OPTS.values())
    _cur_idx = _pkeys.index(st.session_state.llm_provider) if st.session_state.llm_provider in _pkeys else 0

    _sel_label = st.selectbox(
        "LLM Provider",
        options=_plabels,
        index=_cur_idx,
        label_visibility="collapsed",
        help="Choose a model provider. 'Server default' uses the server's configured LLM.",
    )
    _sel_provider = _pkeys[_plabels.index(_sel_label)]

    # Auto-reset dependent fields when provider changes
    if _sel_provider != st.session_state.llm_provider:
        st.session_state.llm_provider    = _sel_provider
        st.session_state.llm_model_inp   = _MODEL_DEFAULTS.get(_sel_provider, "")
        st.session_state.llm_api_key_inp = ""
        st.session_state.llm_base_url_inp = ""
        
        # Persist change
        _set_cookie_js("nrag_llm_provider", _sel_provider)
        _set_cookie_js("nrag_llm_model", st.session_state.llm_model_inp)
        _set_cookie_js("nrag_llm_api_key", "")
        _set_cookie_js("nrag_llm_base_url", "")
        # Removed st.rerun() to ensure JS executes

    if _sel_provider != "server":
        # Model name — pre-filled with provider default, editable
        new_model = st.text_input(
            "Model",
            value=st.session_state.llm_model_inp,
            placeholder=_MODEL_DEFAULTS.get(_sel_provider, "model-name"),
            help="Model identifier (e.g. gpt-4o, llama3). Leave blank to use the provider default.",
        )
        if new_model != st.session_state.llm_model_inp:
            st.session_state.llm_model_inp = new_model
            _set_cookie_js("nrag_llm_model", new_model)

        # Cloud providers: optional API key (falls back to server's key if empty)
        if _sel_provider not in ("ollama",):
            new_llm_key = st.text_input(
                "API Key",
                value=st.session_state.llm_api_key_inp,
                placeholder="Leave blank to use server's key",
                type="password",
                help="Your personal API key. Leave blank to use the key configured on the server.",
            )
            if new_llm_key != st.session_state.llm_api_key_inp:
                st.session_state.llm_api_key_inp = new_llm_key
                _set_cookie_js("nrag_llm_api_key", new_llm_key)

        # Local / custom providers: endpoint URL
        if _sel_provider in ("ollama", "custom"):
            _url_ph = (
                "http://localhost:11434"
                if _sel_provider == "ollama"
                else "https://api.example.com"
            )
            new_llm_url = st.text_input(
                "Endpoint URL",
                value=st.session_state.llm_base_url_inp,
                placeholder=_url_ph,
                help="Base URL of the OpenAI-compatible endpoint (without /v1).",
            )
            if new_llm_url != st.session_state.llm_base_url_inp:
                st.session_state.llm_base_url_inp = new_llm_url
                _set_cookie_js("nrag_llm_base_url", new_llm_url)

        _active_model = (
            st.session_state.llm_model_inp
            or _MODEL_DEFAULTS.get(_sel_provider, "")
        )
        if _active_model:
            st.caption(f"Using **{_active_model}** via {_PROVIDER_OPTS[_sel_provider]}")

    # ── Connection (admin only) ───────────────────────────────────────────────
    if show_admin:
        with st.expander("Connection", expanded=False):
            new_url = st.text_input(
                "API URL", value=st.session_state.api_url,
                placeholder="http://localhost:8058", label_visibility="collapsed",
            )
            if new_url != st.session_state.api_url:
                st.session_state.api_url = new_url.strip()
                _set_cookie_js("nrag_api_url", st.session_state.api_url)
                st.session_state.session_page = 1
                st.session_state._sessions_dirty = True
                st.session_state._health_ts = 0.0
                st.rerun()

            if st.button("⟳  Refresh status", use_container_width=True):
                st.session_state._health_ts = 0.0
                st.session_state.session_page = 1
                st.session_state._sessions_dirty = True
                st.rerun()

    # Finalize any pending cookie updates at the very end of the sidebar 
    # to avoid affecting the layout of components above.
    _render_pending_cookies()


# ── Chat page ─────────────────────────────────────────────────────────────────
if st.session_state.nav == "chat":
    _require_auth()
    sid = st.session_state.current_session_id

    if sid is None:
        st.markdown("## Chat")
        st.info("Create or select a session in the sidebar to start chatting.")
    else:
        # Resolve title
        current_sess = next((s for s in st.session_state.sessions if s["id"] == sid), None)
        title = current_sess["title"] if current_sess else "Chat"

        st.markdown(f"## {title}")
        _model_badge = ""
        if st.session_state.llm_provider != "server":
            _active = (
                st.session_state.get("llm_model_inp")
                or _MODEL_DEFAULTS.get(st.session_state.llm_provider, "")
            )
            if _active:
                _model_badge = f"  ·  model: **{_active}**"
        st.caption(
            f"Session `{sid[:8]}…`  ·  mode: **{st.session_state.search_type}**{_model_badge}"
        )
        st.divider()

        # Render history
        for i, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            avatar = "🧑" if role == "user" else "🤖"
            with st.chat_message(role, avatar=avatar):
                if role != "user" and msg.get("citations"):
                    _render_message_with_citations(
                        msg["content"],
                        msg["citations"],
                        msg.get("message_id") or str(i),
                    )
                else:
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

            # Update session title if generated by backend
            if result.get("title"):
                for s in st.session_state.sessions:
                    if s["id"] == sid:
                        s["title"] = result["title"]
                        break

            st.rerun()


# ── Documents page ────────────────────────────────────────────────────────────
elif st.session_state.nav == "documents":
    _require_auth()
    st.markdown("## Document Library")

    _MIME_ICON = {
        "application/pdf": "📄",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "📝",
        "text/markdown": "📋",
        "text/plain": "📃",
        "text/html": "🌐",
        "image/jpeg": "🖼️",
        "image/png": "🖼️",
        "image/gif": "🎞️",
        "image/webp": "🖼️",
        "image/bmp": "🖼️",
    }

    tab_lib, tab_upload, tab_tasks = st.tabs(["Library", "Upload", "Tasks"])

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
                    dcol, bcol = st.columns([5, 1], vertical_alignment="center")
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
                                st.toast(f"'{doc['title']}' deleted")
                                st.rerun()

    # ── Upload tab ────────────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("Supported formats: **PDF · DOCX · Markdown · TXT · HTML · Images** &nbsp;(max 50 MB each)")

        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 1
        if "upload_results" not in st.session_state:
            st.session_state.upload_results = []

        # Wrap the uploader in a clearable placeholder so it vanishes after ingestion starts
        _uploader_slot = st.empty()
        files = None
        _ingest_btn = False

        with _uploader_slot.container():
            files = st.file_uploader(
                "Drop files here or click to browse",
                type=["pdf", "docx", "md", "txt", "html", "jpg", "jpeg", "png", "gif", "webp", "bmp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key=f"uploader_{st.session_state.uploader_key}"
            )
            if files:
                st.markdown(f"**{len(files)} file(s) selected:**")
                for f in files:
                    st.caption(f"· {f.name}  ({f.size / 1024:.0f} KB)")
                _ingest_btn = st.button("Ingest selected files", type="primary")

        # Show recent upload results if any
        if st.session_state.upload_results and not _ingest_btn:
            st.markdown('<div class="nr-section">Recent Uploads</div>', unsafe_allow_html=True)
            for res in st.session_state.upload_results:
                if res["status"] == "completed":
                    st.success(f"✓ **{res['fname']}** ingested successfully")
                else:
                    st.error(f"✗ {res['fname']}: {res.get('error', 'Failed')}")
            if st.button("Clear History"):
                st.session_state.upload_results = []
                st.rerun()

        if _ingest_btn and files:
            # Clear previous results
            st.session_state.upload_results = []
            _uploader_slot.empty()  # hide the file picker and button

            job_map: Dict[str, str] = {}
            bars: Dict[str, Any] = {}
            hints: Dict[str, Any] = {}

            # ── Upload phase ──────────────────────────────────────────────
            for f in files:
                bar = st.progress(0.0, text=f"Uploading {f.name}…")
                hint = st.empty()
                bars[f.name] = bar
                hints[f.name] = hint
                try:
                    mime = f.type or "application/octet-stream"
                    with httpx.Client(timeout=30) as c:
                        resp = c.post(
                            st.session_state.api_url.rstrip("/") + "/documents/upload",
                            files={"file": (f.name, f.getvalue(), mime)},
                            headers=_hdrs(),
                        )
                        if resp.status_code == 403:
                            try:
                                detail = resp.json().get("detail", "Access denied")
                            except Exception:
                                detail = "Access denied"
                            raise PermissionError(f"Access denied — {detail}. Please contact your administrator.")
                        resp.raise_for_status()
                        jdata = resp.json()
                    job_map[f.name] = jdata.get("job_id", "")
                    bar.progress(0.02, text=f"{f.name} — queued, waiting for worker…")
                except Exception as exc:
                    bar.empty()
                    hint.empty()
                    st.error(f"✗ {f.name}: {exc}")
                    st.session_state.upload_results.append({"fname": f.name, "status": "failed", "error": str(exc)})

            # ── Monitor phase — poll all jobs until complete ───────────────
            pending: Dict[str, str] = {**job_map}
            for _ in range(120):
                if not pending:
                    break
                done_now: List[str] = []
                for fname, jid in pending.items():
                    jr = _api("get", f"/jobs/{jid}", silent=True)
                    if not jr:
                        done_now.append(fname)
                        continue
                    pct = jr.get("progress", 0.0)
                    status = jr.get("status", "pending")
                    bar = bars.get(fname)
                    hint = hints.get(fname)
                    if status == "pending":
                        if bar:
                            bar.progress(0.02, text=f"{fname} — queued…")
                        if hint:
                            if _is_likely_downloading(jr):
                                hint.caption("First run with images — downloading ~1 GB model weights, please wait…")
                            else:
                                hint.empty()
                    elif status == "processing":
                        if bar:
                            bar.progress(max(pct, 0.05), text=f"{fname} — {pct * 100:.0f}%")
                        if hint:
                            if _is_likely_downloading(jr):
                                hint.caption("First run with images — downloading ~1 GB model weights, please wait…")
                            else:
                                hint.empty()
                    elif status == "completed":
                        if bar:
                            bar.progress(1.0, text=f"✓ {fname}")
                        if hint:
                            hint.empty()
                        st.success(f"✓ **{fname}** ingested successfully")
                        st.session_state.upload_results.append({"fname": fname, "status": "completed"})
                        done_now.append(fname)
                    elif status == "failed":
                        if bar:
                            bar.empty()
                        if hint:
                            hint.empty()
                        st.error(f"✗ {fname}: {jr.get('error', 'Processing failed')}")
                        st.session_state.upload_results.append({"fname": fname, "status": "failed", "error": jr.get('error', 'Processing failed')})
                        done_now.append(fname)
                for fname in done_now:
                    pending.pop(fname, None)
                if pending:
                    time.sleep(1)
            
            # Reset UI and show results
            time.sleep(0.5)  # slight pause to let users see 100% complete
            st.session_state.uploader_key += 1
            st.rerun()

    # ── Tasks tab ─────────────────────────────────────────────────────────────
    with tab_tasks:
        _jobs_panel()


# ── Admin page ────────────────────────────────────────────────────────────────
elif st.session_state.nav == "admin":
    st.markdown("## Admin")

    # Verify admin access before rendering any admin UI
    _access_test = _api("get", "/admin/keys", silent=True)
    if _access_test is None:
        if not st.session_state.api_online:
            st.error("Cannot connect to the API — check the Connection settings in the sidebar.")
        else:
            st.error("Admin access denied — your API key does not have admin privileges.")
            st.info(
                "If you are the system owner, bootstrap your first admin key via the CLI:\n\n"
                "```bash\n"
                "curl -X POST http://localhost:8058/admin/keys \\\n"
                "  -H 'Content-Type: application/json' \\\n"
                "  -d '{\"owner\": \"admin\"}'\n"
                "```\n\n"
                "Then paste the returned key in the **Account** section of the sidebar."
            )
        st.stop()

    tab_gen, tab_invites, tab_keys, tab_perms = st.tabs(["Generate Invite", "Invites", "API Keys", "Permissions"])

    # ── Generate invite tab ───────────────────────────────────────────────────
    with tab_gen:
        st.markdown("Create a one-time invite code to give a user access.")
        owner_input = st.text_input("Owner name", placeholder="e.g. alice or team_a", key="inv_owner")
        expiry_days = st.slider("Expires in (days)", min_value=1, max_value=30, value=7)

        if st.button("Generate invite code", type="primary"):
            if not owner_input.strip():
                st.warning("Please enter an owner name.")
            else:
                result = _api(
                    "post", "/admin/invites",
                    json={"owner": owner_input.strip(), "expires_in_days": expiry_days},
                )
                if result:
                    st.session_state._invite_token_result = result
                    st.rerun()

        if st.session_state._invite_token_result:
            inv = st.session_state._invite_token_result
            st.success(f"Invite created for **{inv['owner']}** — expires {inv['expires_at'][:10]}")
            st.markdown("**Share this code with the user:**")
            st.code(inv["token"], language=None)
            if st.button("✓ Done", key="inv_done"):
                st.session_state._invite_token_result = None
                st.rerun()

    # ── Invites list tab ──────────────────────────────────────────────────────
    with tab_invites:
        icol, ricol = st.columns([5, 1])
        with ricol:
            if st.button("⟳ Refresh", key="ref_inv", use_container_width=True):
                st.rerun()

        inv_data = _api("get", "/admin/invites")
        invites = inv_data if isinstance(inv_data, list) else []
        with icol:
            st.markdown(f"**{len(invites)}** invite(s)")

        if not invites:
            st.info("No invites yet.")
        else:
            for inv in invites:
                used = inv.get("used", False)
                owner = inv.get("owner", "")
                token_short = inv.get("token", "")[:12] + "…"
                expires = inv.get("expires_at", "")[:10]
                status_label = "USED" if used else "PENDING"
                status_cls = "nr-completed" if used else "nr-pending"

                row_col, del_col = st.columns([6, 1])
                with row_col:
                    st.markdown(f"""
<div class="nr-job">
  <span class="nr-job-s {status_cls}">{status_label}</span>
  <span style="flex:1;color:#000000;font-weight:700;font-size:0.85rem">{owner}</span>
  <span style="color:#6b7280;font-size:.72rem;font-family:monospace">{token_short}</span>
  <span style="color:#4b5563;font-size:.7rem">exp {expires}</span>
</div>""", unsafe_allow_html=True)
                with del_col:
                    if not used:
                        if st.button("✕", key=f"di_{inv['id']}", help="Revoke"):
                            _api("delete", f"/admin/invites/{inv['id']}", silent=True)
                            st.rerun()

    # ── API Keys list tab ─────────────────────────────────────────────────────
    with tab_keys:
        kcol, rkcol = st.columns([5, 1])
        with rkcol:
            if st.button("⟳ Refresh", key="ref_keys", use_container_width=True):
                st.rerun()

        keys_data = _api("get", "/admin/keys")
        keys = keys_data if isinstance(keys_data, list) else []
        with kcol:
            st.markdown(f"**{len(keys)}** key(s)")

        if not keys:
            st.info("No API keys yet.")
        else:
            for k in keys:
                last_used = (k.get("last_used_at") or "never")[:10]
                created = k.get("created_at", "")[:10]
                row_col, del_col = st.columns([6, 1])
                with row_col:
                    st.markdown(f"""
<div class="nr-job">
  <span style="flex:1;color:#000000;font-weight:700;font-size:0.85rem">{k.get('owner','')}</span>
  <span style="color:#6b7280;font-size:.72rem">created {created}</span>
  <span style="color:#4b5563;font-size:.7rem">last used {last_used}</span>
</div>""", unsafe_allow_html=True)
                with del_col:
                    if st.button("✕", key=f"dk_{k['id']}", help="Revoke key"):
                        _api("delete", f"/admin/keys/{k['id']}", silent=True)
                        st.rerun()

    # ── Permissions tab ───────────────────────────────────────────────────────
    with tab_perms:
        pcol, rpcol = st.columns([5, 1])
        with rpcol:
            if st.button("⟳ Refresh", key="ref_perms", use_container_width=True):
                # Clear cached checkbox states so they re-initialize from API data.
                for _k in list(st.session_state.keys()):
                    if _k.startswith(("p_active_", "p_upload_", "p_delete_", "p_chat_")):
                        del st.session_state[_k]
                st.rerun()

        perms_data = _api("get", "/admin/keys")
        user_keys = [
            k for k in (perms_data if isinstance(perms_data, list) else [])
            if "admin" not in k.get("scopes", [])
        ]
        with pcol:
            st.markdown(f"**{len(user_keys)}** user key(s)")

        if not user_keys:
            st.info("No user keys yet.")
        else:
            h0, h1, h2, h3, h4 = st.columns([3, 1, 1, 1, 1])
            with h0: st.markdown("**Owner**")
            with h1: st.markdown("**Active**")
            with h2: st.markdown("**Upload**")
            with h3: st.markdown("**Delete**")
            with h4: st.markdown("**Chat**")
            st.divider()

            # Define the change handler once outside the loop
            def _on_perm_change(kid: str, owner_name: str):
                # Safely pull current values from session state
                active = st.session_state.get(f"p_active_{kid}")
                upload = st.session_state.get(f"p_upload_{kid}")
                delete = st.session_state.get(f"p_delete_{kid}")
                chat = st.session_state.get(f"p_chat_{kid}")
                
                # Guard: if state is missing for some reason, don't fire API
                if any(v is None for v in [active, upload, delete, chat]):
                    return

                result = _api(
                    "patch",
                    f"/admin/keys/{kid}/permissions",
                    json={
                        "is_active": active,
                        "can_upload": upload,
                        "can_delete": delete,
                        "can_chat": chat,
                    },
                    silent=True,
                )
                if result is not None:
                    st.toast(f"Saved for **{owner_name}**")
                else:
                    st.toast(f"Failed to update **{owner_name}**", icon="❌")

            for k in user_keys:
                kid = k["id"]
                owner_name = k.get("owner", "")

                c0, c1, c2, c3, c4 = st.columns([3, 1, 1, 1, 1])
                with c0:
                    st.markdown(f"**{owner_name}**")
                    st.caption(f"created {k.get('created_at', '')[:10]}")
                with c1:
                    st.checkbox(
                        "Active", value=k.get("is_active", True),
                        key=f"p_active_{kid}", label_visibility="collapsed",
                        on_change=_on_perm_change, args=(kid, owner_name),
                    )
                with c2:
                    st.checkbox(
                        "Upload", value=k.get("can_upload", True),
                        key=f"p_upload_{kid}", label_visibility="collapsed",
                        on_change=_on_perm_change, args=(kid, owner_name),
                    )
                with c3:
                    st.checkbox(
                        "Delete", value=k.get("can_delete", True),
                        key=f"p_delete_{kid}", label_visibility="collapsed",
                        on_change=_on_perm_change, args=(kid, owner_name),
                    )
                with c4:
                    st.checkbox(
                        "Chat", value=k.get("can_chat", True),
                        key=f"p_chat_{kid}", label_visibility="collapsed",
                        on_change=_on_perm_change, args=(kid, owner_name),
                    )
