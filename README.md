# RepoMind

> A Self-Corrective RAG agent that can clone, index, and deeply analyze any GitHub repository — powered by LangGraph, Chroma, and Groq.

---

## Features

- **Clone & Index any GitHub repo** — paste a URL, the agent clones it, indexes the source code with ChromaDB, and generates a persistent metadata summary automatically.
- **Index a local folder** — point the agent at a path on disk to index it without cloning.
- **Self-CRAG (Self-Corrective RAG) pipeline** — after retrieval, a grader LLM filters irrelevant chunks, rewrites the query if needed, generates an answer, and then performs hallucination + usefulness checks (Self-RAG reflection) before returning a response.
- **Planner node** — before retrieval, the agent reads repo metadata and generates an enriched query and analysis plan for more accurate answers.
- **Direct analysis mode** — for structural questions (file trees, code search), the agent skips the vector database and uses LangChain tools directly in a ReAct loop.
- **Human-in-the-loop (HITL) confirmation** — before cloning, the agent pauses and asks the user to confirm. The graph resumes on confirmation or cancels cleanly.
- **Persistent sessions** — conversation threads are stored in Supabase PostgreSQL. Sessions survive server restarts and can be resumed by thread ID.
- **Streaming responses** — the API streams token-by-token output from user-facing graph nodes to the frontend.
- **Dual LLM provider** — tries HuggingFace Inference Providers (free monthly credits) first and falls back to Groq seamlessly.
- **Streamlit + FastAPI UI** — a styled dark-purple web interface for chatting, switching threads, and confirming clone operations.
- **Docker ready** — fully containerised with `Dockerfile.backend`, `Dockerfile.frontend`, and `docker-compose.yml`.

---

## Tech Stack

| Layer | Library / Service |
|---|---|
| **Agent orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) |
| **LLM (primary)** | HuggingFace Inference Providers (`Qwen3-235B`, `Llama-3.3-70B`) |
| **LLM (fallback)** | [Groq](https://groq.com) (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (via HuggingFace Hub) |
| **Vector store** | [ChromaDB](https://www.trychroma.com) (in-process, local disk) |
| **Checkpointing** | `langgraph-checkpoint-postgres` + Supabase PostgreSQL |
| **Chat history** | Supabase PostgreSQL (`chat_messages` table) |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn |
| **Frontend UI** | [Streamlit](https://streamlit.io) |
| **Repo cloning** | [GitPython](https://gitpython.readthedocs.io) |
| **LLM framework** | [LangChain](https://www.langchain.com) |
| **Containerisation** | Docker + Docker Compose |

---

## Graph Architecture

The agent is a stateful LangGraph compiled graph. Every user message flows through the following nodes:

```
START
  └─► classifier   (intent: chat | clone_repo | analyze_repo | index_path)
         │
         ├──[chat]──────────────────────────────────────────────► END
         │
         ├──[clone_repo]──[HITL pause]──[tool_call]──[index_repo]──[chat_update]──► END
         │
         ├──[index_path]──[index_path_tool]──[index_path_update]──────────────────► END
         │
         └──[planner]──[retrieval_router]
                              │
                              ├──[retrieve]──[grade_documents]
                              │                    │
                              │                    ├──(docs found)────► [generate]──[reflect]
                              │                    │                                    │
                              │                    │                         (grounded + useful)──► [finalize]──► END
                              │                    │                         (hallucinated/not useful)──► [rewrite_query]──► [retrieve] (loop)
                              │                    │
                              │                    ├──(no docs, retries left)──► [rewrite_query]──► [retrieve] (loop)
                              │                    └──(max retries)────────────► [fallback]──► END
                              │
                              └──[direct_analyze]──[analyze_tools]──► [direct_analyze] (loop)──► END
```

### Node descriptions

| Node | Description |
|---|---|
| `classifier` | Classifies intent using structured LLM output into one of four intents |
| `chat` | General conversational response (no retrieval) |
| `clone_repo` | Extracts GitHub URL, issues a Human interrupt for confirmation |
| `tool_call` | Executes `clone_github_repo` tool (GitPython) |
| `index_repo` | Parses clone result, indexes source code in Chroma, generates repo metadata |
| `chat_update` | Post-clone confirmation message to the user |
| `index_path` | Extracts a local folder path for indexing |
| `index_path_tool` | Executes `index_path_for_rag` tool |
| `index_path_update` | Parses result and updates metadata |
| `planner` | Reads repo metadata + question → produces enriched query + analysis plan |
| `retrieval_router` | Decides whether vector retrieval is needed (grader LLM) |
| `retrieve` | Fetches top-K chunks from ChromaDB |
| `grade_documents` | Filters irrelevant chunks using grader LLM (CRAG Step 2) |
| `rewrite_query` | Rewrites the query for better retrieval (CRAG Step 3) |
| `generate` | Produces a grounded answer from verified chunks + plan |
| `reflect` | Self-RAG: hallucination check + usefulness check |
| `finalize` | Converts generation to AIMessage and appends to conversation |
| `fallback` | Graceful failure message when max retries are exhausted |
| `direct_analyze` | ReAct loop using `list_repo_files`, `read_repo_file`, `search_repo_code`, `search_repo_rag` |
| `analyze_tools` | ToolNode that executes any tool calls from `direct_analyze` |

---

## Project Structure

```
repo-agent/
├── .env.example               # Template for all required env vars
├── .gitignore
├── Dockerfile.backend         # FastAPI server image
├── Dockerfile.frontend        # Streamlit UI image
├── docker-compose.yml         # Orchestrates both containers
├── requirements.txt
├── pyproject.toml
│
├── api/                       # FastAPI backend
│   ├── server.py              # App factory, lifespan, CORS
│   ├── routes.py              # /api/chat, /api/threads, /api/chat/resume
│   ├── schemas.py             # Pydantic request/response models
│   └── state.py               # Shared graph + checkpointer singletons
│
├── app/                       # Agent core
│   ├── main.py                # CLI entrypoint (for local testing)
│   ├── config.py              # LLM factory (HF → Groq fallback), constants
│   ├── state.py               # AgentState TypedDict
│   ├── prompts.py             # All LLM prompt strings
│   ├── rag.py                 # ChromaDB vector store manager (singleton)
│   ├── tools.py               # LangChain tool definitions
│   ├── metadata.py            # Repo metadata generation + disk persistence
│   ├── chat_store.py          # Postgres chat history (save/get/list)
│   └── graph/
│       ├── builder.py         # create_graph() — wires all nodes and edges
│       ├── schemas.py         # Pydantic models (Intent)
│       ├── edges.py           # All conditional routing functions
│       └── nodes/
│           ├── classifier.py  # Intent classification node
│           ├── chat.py        # General chat node
│           ├── clone.py       # Clone + index + HITL nodes
│           ├── index_path.py  # Local folder indexing nodes
│           ├── planner.py     # Query enrichment + planning node
│           └── crag.py        # Full Self-CRAG pipeline nodes
│
├── frontend/
│   └── app.py                 # Streamlit UI
│
└── tests/
    ├── test_graph_routing.py
    └── test_nodes.py
```

---

## Setup & Running Locally

### Prerequisites
- Python 3.12+
- A [Supabase](https://supabase.com) project (free tier works)
- A [Groq](https://console.groq.com) API key (free)
- Optionally a [HuggingFace](https://huggingface.co/settings/tokens) token with Inference Providers enabled

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and fill in your real values
```

Required variables:

| Variable | Description |
|---|---|
| `groq_api_key` | Groq API key |
| `Hf_token` | HuggingFace token (optional, falls back to Groq) |
| `SUPABASE_DB_URI` | Full PostgreSQL connection string from Supabase |

### 3. Start the backend API

```bash
# From the repo-agent/ directory
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Wait until the terminal prints `[*] Agent ready - accepting requests.`

### 4. Start the frontend

Open a second terminal:

```bash
streamlit run frontend/app.py
```

Navigate to `http://localhost:8501` in your browser.

### CLI mode (no UI)

```bash
python -m app.main
```

---

## Deploy with Docker

```bash
# Copy and fill in the env file
cp .env.example .env

# Build and start both services
docker compose up --build

# Open http://localhost:8501
```

The `docker-compose.yml` mounts two named volumes (`chroma_data`, `repo_data`) so your vector indices and cloned repos persist across container restarts.

---

## Deploying to a Cloud Provider

1. Push `repo-agent/` to a GitHub repository.
2. Create **two services** on your cloud provider (e.g. Render, Railway, Fly.io):
   - **Backend**: build from `Dockerfile.backend`, expose port `8000`.
   - **Frontend**: build from `Dockerfile.frontend`, expose port `8501`.
3. Add all variables from `.env.example` to each service's secret configuration.
4. Set `API_BASE` on the **frontend** service to the public URL of your backend.
5. Set `CORS_ORIGINS` on the **backend** service to the public URL of your frontend.

---

## How Sessions Work

Every conversation is identified by a `thread_id` (UUID). The LangGraph `PostgresSaver` checkpointer stores the full agent state in Supabase after every node execution. A separate `chat_messages` table stores the human-readable message history.

- **Resuming a session**: send any subsequent message with the same `thread_id` — the agent picks up exactly where it left off, including any pending HITL interrupt.
- **New session**: send a message without a `thread_id` (the API creates one and returns it).

---

## How to Reuse This

This project is structured to be easily extended:

- **Add a new intent**: add a branch to `classifier.py`, register a new route value in `edges.py:route_intent`, add a node in `nodes/`, and wire it in `builder.py`.
- **Swap the LLM**: change `config.py:build_llms()` — everything else stays the same.
- **Swap the vector store**: replace `rag.py` with a different backend (Pinecone, Qdrant, etc.) — the `search_repo_rag` tool and `retrieve_node` just call `rag_manager`.
- **Swap the database**: `chat_store.py` has a clean 4-function interface (`init_db`, `save_message`, `get_history`, `list_threads`). Swap the `psycopg` calls for any other backend.
- **Add a new tool**: define it in `tools.py`, add it to the `analyze_tools` list in `builder.py`.

---

## License

MIT
