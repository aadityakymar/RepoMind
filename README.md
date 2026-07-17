# Repo Agent

A Self-Corrective RAG (Self-CRAG) agent for exploring GitHub repositories, powered by LangGraph.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and fill in your API keys

# Run the agent
python -m app.main
```

## Architecture

```
app/
├── main.py          CLI entrypoint
├── config.py        Model names, MAX_RETRIES, LLM factory
├── state.py         AgentState TypedDict
├── prompts.py       All LLM prompts
├── rag.py           In-memory Chroma vector store manager
├── tools.py         LangChain tools (clone, read, search, index)
├── metadata.py      Repo metadata generation + persistence
├── chat_store.py    SQLite chat history logger (for frontend)
└── graph/
    ├── builder.py   create_graph() — wires nodes + edges
    ├── schemas.py   Pydantic models (Intent)
    ├── edges.py     All conditional routing functions
    └── nodes/
        ├── classifier.py
        ├── chat.py
        ├── clone.py
        ├── index_path.py
        ├── planner.py
        └── crag.py  Self-CRAG + direct-analyze nodes
```

## Chat History

Every conversation is saved to `chat_history.db` (SQLite) keyed by `thread_id`.
The `api/` folder is where the future frontend server will live.
