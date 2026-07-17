"""
app/graph/builder.py — Graph factory and architecture visualiser.

create_graph(checkpointer=None)
    Wires all nodes and edges together and returns a compiled LangGraph.
    The checkpointer argument is optional (used for persistent sessions).

print_graph_architecture()
    Prints a human-readable summary of all nodes and edges.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.state import AgentState
from app.config import build_llms
from app.tools import (
    clone_github_repo,
    list_repo_files,
    read_repo_file,
    search_repo_code,
    search_repo_rag,
    index_path_for_rag,
)
from app.graph.edges import (
    route_intent,
    router_condition,
    grade_condition,
    analyze_tools_condition,
    clone_proceed_condition,
    make_reflect_condition,
)
from app.graph.nodes.classifier import make_classifier_node
from app.graph.nodes.chat       import make_chat_node
from app.graph.nodes.clone      import (
    make_clone_repo_node,
    make_index_repo_node,
    make_chat_update_node,
)
from app.graph.nodes.index_path import (
    make_index_path_node,
    make_index_path_update_node,
)
from app.graph.nodes.planner import make_planner_node
from app.graph.nodes.crag    import (
    make_retrieval_router_node,
    make_retrieve_node,
    make_grade_documents_node,
    make_rewrite_query_node,
    make_generate_node,
    reflect_node,
    make_finalize_node,
    make_fallback_node,
    make_direct_analyze_node,
)


# ---------------------------------------------------------------------------
# Graph Factory
# ---------------------------------------------------------------------------
def create_graph(checkpointer=None):
    """
    Builds and compiles the LangGraph agent with:
      - Explicit intent classification (chat / clone_repo / analyze_repo / index_path)
      - Planner Node: analyses query + repo metadata, produces enriched query + plan
      - Self-Corrective RAG (Self-CRAG) for repository analysis
      - Persistent repo metadata (generated at clone time, loaded each turn)
      - Lazy chunk summarisation (summaries added to metadata on first retrieval)
      - Session-scoped vector store (cleared on exit)

    Graph flow:
      START → classifier
        → chat          → END
        → clone_repo    → tool_call → index_repo → chat_update → END
        → planner       → retrieval_router
            → retrieve  → grade_documents → generate → reflect → finalize → END
                                         ↘ rewrite_query ↗
            → direct_analyze → [analyze_tools loop] → END
        → index_path    → index_path_tool → index_path_update → END
    """
    # ── LLM initialisation (HF Inference Providers → Groq fallback) ─────────
    llm, grader_llm = build_llms()

    analyze_tools = [list_repo_files, read_repo_file, search_repo_code, search_repo_rag]

    # ── Instantiate nodes from factories ────────────────────────────────────
    classifier_node       = make_classifier_node(llm)
    chat_node             = make_chat_node(llm)
    clone_repo_node       = make_clone_repo_node(llm)
    index_repo_node       = make_index_repo_node(llm)
    chat_update_node      = make_chat_update_node(llm)
    index_path_node       = make_index_path_node(llm)
    index_path_update_node = make_index_path_update_node(llm)
    planner_node          = make_planner_node(llm)
    retrieval_router_node = make_retrieval_router_node(grader_llm)
    retrieve_node         = make_retrieve_node()
    grade_documents_node  = make_grade_documents_node(grader_llm)
    rewrite_query_node    = make_rewrite_query_node(grader_llm)
    generate_node         = make_generate_node(llm)
    finalize_node         = make_finalize_node()
    fallback_node         = make_fallback_node(llm)
    direct_analyze_node   = make_direct_analyze_node(llm, analyze_tools)
    reflect_condition     = make_reflect_condition(grader_llm)

    # ── Build graph ──────────────────────────────────────────────────────────
    builder = StateGraph(AgentState)

    # --- Shared nodes ---
    builder.add_node("classifier",  classifier_node)
    builder.add_node("chat",        chat_node)
    builder.add_node("clone_repo",  clone_repo_node)
    builder.add_node("tool_call",   ToolNode(tools=[clone_github_repo]))
    builder.add_node("index_repo",  index_repo_node)
    builder.add_node("chat_update", chat_update_node)

    # --- Index path nodes ---
    builder.add_node("index_path",        index_path_node)
    builder.add_node("index_path_tool",   ToolNode(tools=[index_path_for_rag]))
    builder.add_node("index_path_update", index_path_update_node)

    # --- Planner node ---
    builder.add_node("planner", planner_node)

    # --- Self-CRAG pipeline nodes ---
    builder.add_node("retrieval_router", retrieval_router_node)
    builder.add_node("retrieve",         retrieve_node)
    builder.add_node("grade_documents",  grade_documents_node)
    builder.add_node("rewrite_query",    rewrite_query_node)
    builder.add_node("generate",         generate_node)
    builder.add_node("reflect",          reflect_node)
    builder.add_node("finalize",         finalize_node)
    builder.add_node("fallback",         fallback_node)

    # --- Direct analyze (no RAG) ---
    builder.add_node("direct_analyze", direct_analyze_node)
    builder.add_node("analyze_tools",  ToolNode(tools=analyze_tools))

    # ----- Entry -----
    builder.add_edge(START, "classifier")

    # ----- Classifier → branches -----
    builder.add_conditional_edges(
        "classifier",
        route_intent,
        {
            "clone_repo":   "clone_repo",
            "analyze_repo": "planner",
            "index_path":   "index_path",
            "chat":         "chat",
        },
    )

    # ----- Chat branch -----
    builder.add_edge("chat", END)

    # ----- Clone branch: clone → (confirm?) → tool_call → index → chat_update → END -----
    builder.add_conditional_edges(
        "clone_repo",
        clone_proceed_condition,
        {
            "proceed":   "tool_call",
            "cancelled": END,
        },
    )
    builder.add_edge("tool_call",   "index_repo")
    builder.add_edge("index_repo",  "chat_update")
    builder.add_edge("chat_update", END)

    # ----- Index path branch -----
    builder.add_edge("index_path",        "index_path_tool")
    builder.add_edge("index_path_tool",   "index_path_update")
    builder.add_edge("index_path_update", END)

    # ----- Planner → Retrieval Router -----
    builder.add_edge("planner", "retrieval_router")

    # ----- Self-CRAG pipeline -----
    builder.add_conditional_edges(
        "retrieval_router",
        router_condition,
        {
            "retrieve":       "retrieve",
            "direct_analyze": "direct_analyze",
        },
    )
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        grade_condition,
        {
            "generate": "generate",
            "rewrite":  "rewrite_query",
            "fallback": "fallback",
        },
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate", "reflect")
    builder.add_conditional_edges(
        "reflect",
        reflect_condition,
        {
            "finalize": "finalize",
            "rewrite":  "rewrite_query",
        },
    )
    builder.add_edge("finalize", END)
    builder.add_edge("fallback", END)

    # ----- Direct analyze branch (tools loop) -----
    builder.add_conditional_edges(
        "direct_analyze",
        analyze_tools_condition,
        {
            "use_tools": "analyze_tools",
            "done":      END,
        },
    )
    builder.add_edge("analyze_tools", "direct_analyze")

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Architecture Visualiser
# ---------------------------------------------------------------------------
def print_graph_architecture() -> None:
    """
    Prints a human-readable summary of every node and edge in the compiled
    agent graph.
    """
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RESET  = "\033[0m"

    nodes = [
        ("START",              "Entry point — LangGraph pseudo-node"),
        ("classifier",         "Classifies user intent: chat | clone_repo | analyze_repo | index_path"),
        ("chat",               "General conversational response (LLM)"),
        ("clone_repo",         "Extracts GitHub URL and forces clone_github_repo tool call"),
        ("tool_call",          "ToolNode — executes clone_github_repo"),
        ("index_repo",         "Parses clone result, indexes repo in RAG, generates metadata"),
        ("chat_update",        "Post-clone friendly summary to the user"),
        ("index_path",         "Extracts target path and forces index_path_for_rag tool call"),
        ("index_path_tool",    "ToolNode — executes index_path_for_rag"),
        ("index_path_update",  "Parses index result, updates metadata, emits confirmation"),
        ("planner",            "Analyses question + repo metadata → plan, planner_query, tool_hints"),
        ("retrieval_router",   "Decides: does this query need vector DB retrieval? (grader LLM)"),
        ("retrieve",           "Fetches top-K chunks from the vector store (CRAG Step 1)"),
        ("grade_documents",    "Filters irrelevant chunks with grader LLM (CRAG Step 2)"),
        ("rewrite_query",      "Rewrites query for better retrieval (CRAG Step 3)"),
        ("generate",           "Produces grounded answer from verified chunks + plan (Step 4)"),
        ("reflect",            "Self-RAG: hallucination + usefulness checks (Step 5)"),
        ("finalize",           "Converts generation string → AIMessage, appends to history"),
        ("fallback",           "Max-retries exhausted — tells user & suggests alternatives"),
        ("direct_analyze",     "Tool-based analysis when RAG is skipped (LLM + tools)"),
        ("analyze_tools",      "ToolNode — executes list/read/search/rag tools"),
        ("END",                "Exit point — LangGraph pseudo-node"),
    ]

    edges = [
        ("START",             "classifier",        None),
        ("classifier",        "chat",              "intent=chat"),
        ("classifier",        "clone_repo",        "intent=clone_repo"),
        ("classifier",        "planner",           "intent=analyze_repo"),
        ("classifier",        "index_path",        "intent=index_path"),
        ("chat",              "END",               None),
        ("clone_repo",        "tool_call",         None),
        ("tool_call",         "index_repo",        None),
        ("index_repo",        "chat_update",       None),
        ("chat_update",       "END",               None),
        ("index_path",        "index_path_tool",   None),
        ("index_path_tool",   "index_path_update", None),
        ("index_path_update", "END",               None),
        ("planner",           "retrieval_router",  None),
        ("retrieval_router",  "retrieve",          "retrieval_needed=True"),
        ("retrieval_router",  "direct_analyze",    "retrieval_needed=False"),
        ("retrieve",          "grade_documents",   None),
        ("grade_documents",   "generate",          "docs found"),
        ("grade_documents",   "rewrite_query",     "no docs, retries left"),
        ("grade_documents",   "fallback",          "max retries"),
        ("rewrite_query",     "retrieve",          "retry loop"),
        ("generate",          "reflect",           None),
        ("reflect",           "finalize",          "grounded & useful"),
        ("reflect",           "rewrite_query",     "hallucinated / not useful"),
        ("finalize",          "END",               None),
        ("fallback",          "END",               None),
        ("direct_analyze",    "analyze_tools",     "tool_calls present"),
        ("direct_analyze",    "END",               "done"),
        ("analyze_tools",     "direct_analyze",    "tool loop"),
    ]

    width = 72
    print()
    print(BOLD + "┌" + "─" * width + "┐" + RESET)
    print(BOLD + "│{:^{w}}│".format("  AGENT GRAPH ARCHITECTURE", w=width) + RESET)
    print(BOLD + "└" + "─" * width + "┘" + RESET)

    print()
    print(CYAN + BOLD + f"  {'NODES':─<{width - 2}}" + RESET)
    for name, desc in nodes:
        tag  = f"  [{name}]"
        line = f"{tag:<26}  {desc}"
        print(GREEN + line + RESET)

    print()
    print(CYAN + BOLD + f"  {'EDGES':─<{width - 2}}" + RESET)
    for src, dst, label in edges:
        arrow = f"  {src:<22} ──▶  {dst}"
        if label:
            arrow += f"   ({YELLOW}{label}{RESET})"
        print(arrow)

    print()
    print(BOLD + "─" * (width + 2) + RESET)
    print()
