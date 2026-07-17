"""
api/server.py — FastAPI application factory and lifespan.

Run with:
    cd repo-agent
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env before importing anything from app/ (config.py reads env vars)
load_dotenv()


# ---------------------------------------------------------------------------
# Lifespan — initialise expensive globals once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      1. Connect to Supabase Postgres and set up LangGraph checkpoint tables.
      2. Build the LangGraph agent (loads LLMs + embedding model).
      3. Expose graph + checkpointer via api.state.
    Shutdown:
      Clear the in-memory vector store.
    """
    print("\n[*] Starting Repo Agent API...")

    from langgraph.checkpoint.postgres import PostgresSaver
    from app.chat_store import init_db
    from app.graph import create_graph
    from api import state as api_state

    db_uri = os.environ.get("SUPABASE_DB_URI")
    if not db_uri:
        raise EnvironmentError(
            "'SUPABASE_DB_URI' is not set in your environment. "
            "Add it to .env before starting the server."
        )

    # Initialise our chat_messages Postgres table
    init_db()

    # PostgresSaver creates its own checkpoint tables automatically via setup()
    print("[*] Connecting to Supabase Postgres for LangGraph checkpointer...")
    checkpointer = PostgresSaver.from_conn_string(db_uri)
    checkpointer.setup()          # idempotent: creates tables if they don't exist
    print("[*] Checkpointer ready.")

    graph = create_graph(checkpointer=checkpointer)

    api_state.set_checkpointer(checkpointer)
    api_state.set_graph(graph)

    print("[*] Agent ready - accepting requests.\n")

    yield  # server runs here

    # Teardown
    from app.rag import rag_manager
    rag_manager.clear()
    print("[*] Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
# CORS origins: allow localhost dev + any production frontend domain.
# In production, replace "*" with your actual Streamlit Cloud / frontend URL.
_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501",
).split(",")

app = FastAPI(
    title="Repo Agent API",
    description="LangGraph-powered GitHub repository analysis agent.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and mount router AFTER app is created (avoids circular imports)
from api.routes import router  # noqa: E402
app.include_router(router, prefix="/api")


@app.get("/health")
async def health():
    from api.state import get_graph
    return {"status": "ok", "agent_ready": get_graph() is not None}
