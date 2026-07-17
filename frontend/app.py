"""
frontend/app.py — Streamlit UI for the Repo Agent.

Run with (from the repo-agent/ directory):
    streamlit run frontend/app.py

The FastAPI backend must be running on http://localhost:8000.
"""

import os
import re
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
API_BASE    = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")
PAGE_BG     = "#1B0055"      # deep violet — matches the provided design
SIDEBAR_BG  = "#120038"
CARD_BG     = "#d4d4d4"
CARD_ACTIVE = "#f0f0f0"

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Repo Agent",
    page_icon   = "🤖",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────────────────────────────────────
def _css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,600;0,700;1,900&display=swap');

    /* ── Global ─────────────────────────────────────────────────────────── */
    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', system-ui, sans-serif !important;
        background-color: {PAGE_BG} !important;
        color: #ffffff;
    }}

    /* Hide Streamlit chrome */
    header[data-testid="stHeader"]  {{ display: none !important; }}
    #MainMenu                        {{ display: none !important; }}
    footer                           {{ display: none !important; }}
    [data-testid="stToolbar"]        {{ display: none !important; }}

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG} !important;
        border-right: 1px solid #2d0d70 !important;
    }}
    [data-testid="stSidebar"] * {{
        color: #ffffff !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.15) !important;
    }}

    /* Sidebar buttons → pill style */
    [data-testid="stSidebar"] .stButton > button {{
        background-color: {CARD_BG} !important;
        color: #1a1a1a !important;
        border: none !important;
        border-radius: 25px !important;
        width: 100% !important;
        padding: 0.45rem 1rem !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.35rem !important;
        transition: background-color 0.15s, transform 0.1s !important;
        text-align: left !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: #bdbdbd !important;
        transform: translateX(3px) !important;
    }}

    /* ── Main content area ───────────────────────────────────────────────── */
    .main .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 1100px;
    }}

    /* Welcome title */
    .welcome-title {{
        text-align: center;
        font-size: 5rem;
        font-weight: 900;
        font-style: italic;
        color: #ffffff;
        letter-spacing: 0.06em;
        margin: 1.5rem 0 2.5rem 0;
        text-shadow: 0 0 60px rgba(160, 80, 255, 0.5);
    }}

    /* ── Feature cards  (all main-area buttons sized as cards) ───────────── */
    [data-testid="stMainBlockContainer"] .stButton > button {{
        height: 190px !important;
        border-radius: 20px !important;
        background-color: {CARD_BG} !important;
        color: #2a2a2a !important;
        border: none !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        white-space: normal !important;
        padding: 1.2rem !important;
        line-height: 1.5 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stMainBlockContainer"] .stButton > button:hover {{
        background-color: {CARD_ACTIVE} !important;
        transform: translateY(-5px) !important;
        box-shadow: 0 10px 35px rgba(0,0,0,0.55) !important;
    }}

    /* Override card height for action buttons (proceed / cancel) */
    button[kind="primary"] {{
        height: auto !important;
        border-radius: 10px !important;
        background-color: #6d28d9 !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 3px 12px rgba(109,40,217,0.5) !important;
    }}
    button[kind="primary"]:hover {{
        background-color: #7c3aed !important;
        transform: none !important;
    }}
    button[kind="secondary"] {{
        height: auto !important;
        border-radius: 10px !important;
        background-color: rgba(255,255,255,0.12) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
    }}
    button[kind="secondary"]:hover {{
        background-color: rgba(255,255,255,0.2) !important;
        transform: none !important;
    }}

    /* ── Chat messages ───────────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {{
        background-color: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 14px !important;
        margin-bottom: 0.6rem !important;
        padding: 0.75rem 1rem !important;
    }}
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] code {{
        color: #f0e6ff !important;
    }}
    [data-testid="stChatMessage"] pre {{
        background-color: rgba(0,0,0,0.3) !important;
        border-radius: 8px !important;
    }}

    /* ── Chat input ──────────────────────────────────────────────────────── */
    [data-testid="stChatInput"] > div {{
        background-color: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        border-radius: 28px !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: #ffffff !important;
        background-color: transparent !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: rgba(255,255,255,0.45) !important;
    }}
    [data-testid="stChatInput"] button {{
        background-color: #6d28d9 !important;
        border-radius: 50% !important;
    }}

    /* ── Interrupt confirmation box ──────────────────────────────────────── */
    .interrupt-box {{
        background-color: rgba(255, 190, 0, 0.08);
        border: 1px solid rgba(255, 190, 0, 0.45);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        color: #ffe066;
        font-size: 0.95rem;
        line-height: 1.6;
    }}
    .interrupt-url {{
        color: #82e0ff;
        font-weight: 700;
        font-size: 1.05rem;
    }}

    /* ── Spinner ─────────────────────────────────────────────────────────── */
    .stSpinner > div > div {{
        border-top-color: #9c60ff !important;
    }}

    /* ── Dividers ────────────────────────────────────────────────────────── */
    hr {{ border-color: rgba(255,255,255,0.12) !important; }}

    /* ── Streamlit scrollbar ────────────────────────────────────────────── */
    ::-webkit-scrollbar       {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {SIDEBAR_BG}; }}
    ::-webkit-scrollbar-thumb {{ background: #4a1f9e; border-radius: 3px; }}
    </style>
    """, unsafe_allow_html=True)

_css()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _strip_ansi(text: str) -> str:
    """Remove ANSI terminal colour codes produced by the CLI interrupt prompt."""
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _api_get(path: str, default=None):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=8)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return default
    except Exception:
        return default


def _api_post(path: str, payload: dict | None = None) -> tuple:
    """Returns (data, error_str). error_str is None on success."""
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=180)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API server. Please start it with:\n`python -m uvicorn api.server:app --port 8000`"
    except Exception as e:
        return None, f"API error: {e}"


def _check_api() -> bool:
    """Return True if the API health endpoint responds OK."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "thread_id":        None,   # active thread UUID (str) or None
    "messages":         [],     # list[{role, content, timestamp}]
    "interrupted":      False,  # graph paused for clone confirmation
    "interrupt_prompt": "",     # raw prompt string from the graph
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# API health check  (bail early with a clear message if backend is down)
# ─────────────────────────────────────────────────────────────────────────────
_api_ok = _check_api()
if not _api_ok:
    st.markdown("""
    <div style="
        background: rgba(220,50,50,0.15);
        border: 1px solid rgba(220,50,50,0.55);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        color: #ff9090;
        font-size: 0.95rem;
        margin: 2rem auto;
        max-width: 700px;
    ">
        <strong>API server is not running</strong><br><br>
        Open a second terminal, activate your venv, then run:<br><br>
        <code style="
            display: block;
            background: rgba(0,0,0,0.35);
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 0.9rem;
            margin-top: 6px;
        ">
            python -m uvicorn api.server:app --port 8000
        </code><br>
        Refresh this page once it prints <em>Application startup complete</em>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()   # halt rendering — no sidebar, no cards, no errors


# ─────────────────────────────────────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────────────────────────────────────
def _new_thread() -> str | None:
    data, err = _api_post("/api/threads")
    if err:
        st.error(err)
        return None
    return data["thread_id"]


def _send_message(thread_id: str, message: str):
    """Post a message to the API and update session state with the response."""
    st.session_state.messages.append({"role": "user", "content": message, "timestamp": ""})

    with st.spinner("Thinking..."):
        data, err = _api_post("/api/chat", {"thread_id": thread_id, "message": message})

    if err:
        st.error(err)
        # Roll back the optimistically added user message
        st.session_state.messages.pop()
        return

    if data["interrupted"]:
        st.session_state.interrupted      = True
        st.session_state.interrupt_prompt = data.get("interrupt_prompt", "")
    else:
        st.session_state.interrupted = False
        if data["response"]:
            st.session_state.messages.append({
                "role": "assistant", "content": data["response"], "timestamp": ""
            })


def _resume(thread_id: str, answer: str):
    """Resume a paused graph after the user confirms or cancels the clone."""
    st.session_state.interrupted      = False
    st.session_state.interrupt_prompt = ""

    with st.spinner("Processing..."):
        data, err = _api_post("/api/chat/resume", {"thread_id": thread_id, "answer": answer})

    if err:
        st.error(err)
        return

    if data and data["response"]:
        st.session_state.messages.append({
            "role": "assistant", "content": data["response"], "timestamp": ""
        })


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Repo Agent")
    st.markdown("---")

    # New Chat button
    if st.button("＋  New Chat", key="btn_new_chat"):
        tid = _new_thread()
        if tid:
            st.session_state.thread_id        = tid
            st.session_state.messages         = []
            st.session_state.interrupted      = False
            st.session_state.interrupt_prompt = ""
            st.rerun()

    st.markdown("---")

    # Thread history list
    threads = _api_get("/api/threads", default=[])
    if threads:
        st.markdown("**Recent Chats**")
        for t in threads[:12]:
            tid      = t["thread_id"]
            short    = tid[:8]
            active   = tid == st.session_state.thread_id
            label    = f"{'▶  ' if active else ''}{short}…"
            if st.button(label, key=f"thread_{tid}"):
                if not active:
                    history = _api_get(f"/api/threads/{tid}/history", default=[])
                    st.session_state.thread_id        = tid
                    st.session_state.messages         = history or []
                    st.session_state.interrupted      = False
                    st.session_state.interrupt_prompt = ""
                    st.rerun()
    else:
        st.caption("No chats yet — start a new one.")

# ─────────────────────────────────────────────────────────────────────────────
# Main area — Welcome screen
# ─────────────────────────────────────────────────────────────────────────────
CARDS = [
    ("💬", "chat node",                "Start a general conversation with the agent."),
    ("🗂️",  "clone repo",               "Clone a GitHub repository and index it for RAG."),
    ("🔍", "ask from specific folder", "Run RAG on a local folder or file path."),
    ("🔬", "analyse a repo/folder",    "Deep-dive into a cloned or local repository."),
]

if st.session_state.thread_id is None:
    # Title
    st.markdown('<div class="welcome-title">WELCOME</div>', unsafe_allow_html=True)

    # 4 capability cards
    cols = st.columns(4, gap="large")
    for col, (icon, title, _desc) in zip(cols, CARDS):
        with col:
            if st.button(
                f"{icon}\n\n**{title}**",
                key=f"card_{title}",
                use_container_width=True,
                help=_desc,
            ):
                tid = _new_thread()
                if tid:
                    st.session_state.thread_id = tid
                    st.session_state.messages  = []
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat input visible on the welcome screen too
    if welcome_prompt := st.chat_input("Type here…", key="welcome_input"):
        tid = _new_thread()
        if tid:
            st.session_state.thread_id = tid
            st.session_state.messages  = []
            _send_message(tid, welcome_prompt)
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Main area — Active Chat screen
# ─────────────────────────────────────────────────────────────────────────────
else:
    # Message history
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))

    # ── Interrupt: clone confirmation ─────────────────────────────────────
    if st.session_state.interrupted:
        clean = _strip_ansi(st.session_state.interrupt_prompt)

        # Extract URL for highlighted display
        url_match = re.search(r"https?://[^\s\n]+", clean)
        url_line  = f'<span class="interrupt-url">→ {url_match.group()}</span><br>' if url_match else ""
        # Strip the URL line from the body (we'll render it separately)
        body = re.sub(r"https?://[^\s\n]+", "", clean).strip()

        st.markdown(f"""
        <div class="interrupt-box">
            <strong>⚡ Confirmation Required</strong><br><br>
            {url_line}
            {body.replace(chr(10), "<br>")}
        </div>
        """, unsafe_allow_html=True)

        c1, c2, _ = st.columns([1, 1, 5])
        with c1:
            if st.button("✅ Yes, clone it", type="primary", use_container_width=True, key="btn_yes"):
                _resume(st.session_state.thread_id, "yes")
                st.rerun()
        with c2:
            if st.button("❌ Cancel", type="secondary", use_container_width=True, key="btn_no"):
                _resume(st.session_state.thread_id, "no")
                st.rerun()

    # ── Normal chat input ─────────────────────────────────────────────────
    else:
        if prompt := st.chat_input("Type here…", key="chat_input"):
            _send_message(st.session_state.thread_id, prompt)
            st.rerun()
