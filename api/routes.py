"""
api/routes.py — All FastAPI route handlers.

Endpoints:
    POST   /api/threads               Create a new thread_id
    GET    /api/threads               List all threads (from SQLite)
    GET    /api/threads/{id}/history  Get message history for a thread
    POST   /api/chat                  Send a message; may return interrupted=True
    POST   /api/chat/resume           Resume a paused graph (clone confirmation)
"""

import uuid
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from api.schemas import ChatRequest, ChatResponse, ResumeRequest, NewThreadResponse
from api.state import get_graph
from app.chat_store import save_message, get_history, list_threads

router = APIRouter()

# Nodes whose LLM output is user-facing — same set as the CLI
_USER_FACING_NODES = {
    "chat",
    "chat_update",
    "generate",
    "fallback",
    "direct_analyze",
    "index_path_update",
    "clone_repo",
}


# ---------------------------------------------------------------------------
# Internal helper — run the graph and collect response / interrupt info
# ---------------------------------------------------------------------------
def _run_graph(graph, input_data, config: dict) -> tuple[str, bool, str]:
    """
    Streams the graph until it finishes or hits an interrupt.

    Returns:
        response_text    — accumulated user-facing text
        interrupted      — True if the graph paused for human confirmation
        interrupt_prompt — the message to show the user when interrupted
    """
    response_text = ""

    for mode, data in graph.stream(
        input_data,
        config=config,
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            chunk, metadata = data
            node = metadata.get("langgraph_node", "")
            if node in _USER_FACING_NODES:
                content     = getattr(chunk, "content", "")
                tool_chunks = getattr(chunk, "tool_call_chunks", None)
                if content and not tool_chunks:
                    response_text += content

    # Check whether the graph paused for human-in-the-loop
    graph_state = graph.get_state(config)
    interrupted = bool(
        graph_state.tasks
        and any(getattr(t, "interrupts", None) for t in graph_state.tasks)
    )
    interrupt_prompt = ""
    if interrupted:
        for task in graph_state.tasks:
            ivs = getattr(task, "interrupts", None)
            if ivs:
                interrupt_prompt = ivs[0].value
                break

    return response_text, interrupted, interrupt_prompt


# ---------------------------------------------------------------------------
# Thread management
# ---------------------------------------------------------------------------
@router.post("/threads", response_model=NewThreadResponse)
async def create_thread():
    """Create a new session thread and return its UUID."""
    return {"thread_id": str(uuid.uuid4())}


@router.get("/threads")
async def get_threads():
    """Return all threads ordered by most recently active."""
    return list_threads()


@router.get("/threads/{thread_id}/history")
async def thread_history(thread_id: str):
    """Return the full message history for a thread (oldest first)."""
    return get_history(thread_id)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Process one user turn.

    - Saves the user message to SQLite.
    - Fetches any existing in-memory state for the thread (repo_name / repo_metadata).
    - Runs the LangGraph agent.
    - If the graph pauses at the clone-confirmation interrupt, returns
      interrupted=True with the prompt but does NOT save the (missing)
      assistant response yet.
    - On a normal completion, saves the assistant response and returns it.
    """
    graph = get_graph()
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not ready — try again in a moment.")

    config = {"configurable": {"thread_id": req.thread_id}}

    save_message(req.thread_id, "user", req.message)

    # Pull repo context from the MemorySaver state (if this thread already ran before)
    current = graph.get_state(config)
    repo_name:     str  = ""
    repo_metadata: dict = {}
    if current and current.values:
        repo_name     = current.values.get("repo_name", "")
        repo_metadata = current.values.get("repo_metadata", {})

    initial_state = {
        "messages":         [HumanMessage(content=req.message)],
        "intent":           "",
        "repo_name":        repo_name,
        "repo_metadata":    repo_metadata,
        "plan":             "",
        "planner_query":    "",
        "tool_hints":       [],
        "question":         req.message,
        "documents":        [],
        "generation":       "",
        "loop_count":       0,
        "retrieval_needed": True,
    }

    response_text, interrupted, interrupt_prompt = _run_graph(graph, initial_state, config)

    if not interrupted and response_text:
        save_message(req.thread_id, "assistant", response_text)

    return ChatResponse(
        thread_id=req.thread_id,
        response=response_text,
        interrupted=interrupted,
        interrupt_prompt=interrupt_prompt if interrupted else None,
    )


# ---------------------------------------------------------------------------
# Resume (human-in-the-loop)
# ---------------------------------------------------------------------------
@router.post("/chat/resume", response_model=ChatResponse)
async def resume_chat(req: ResumeRequest):
    """
    Resume a paused LangGraph execution after the user answers the
    clone-confirmation prompt with 'yes' or 'no'.
    """
    graph = get_graph()
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not ready.")

    config = {"configurable": {"thread_id": req.thread_id}}

    response_text, interrupted, interrupt_prompt = _run_graph(
        graph, Command(resume=req.answer), config
    )

    if response_text:
        save_message(req.thread_id, "assistant", response_text)

    return ChatResponse(
        thread_id=req.thread_id,
        response=response_text,
        interrupted=interrupted,
        interrupt_prompt=interrupt_prompt if interrupted else None,
    )
