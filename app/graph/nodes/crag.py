"""
app/graph/nodes/crag.py — Self-CRAG and direct-analyze pipeline nodes.

Nodes (all factory functions unless otherwise noted):
    make_retrieval_router_node  → Step 0: decide if RAG retrieval is needed
    make_retrieve_node          → Step 1: fetch top-K chunks
    make_grade_documents_node   → Step 2 (CRAG): filter irrelevant chunks
    make_rewrite_query_node     → Step 3 (CRAG): improve query for retry
    make_generate_node          → Step 4: produce grounded answer
    reflect_node                → Step 5 (Self-RAG): passthrough (routing in edges.py)
    make_finalize_node          → push generation into messages
    make_fallback_node          → max-retries exhausted message
    make_direct_analyze_node    → tool-based analysis (no RAG)
"""

import json

from langchain_core.messages import HumanMessage, AIMessage

from app.state import AgentState
from app.rag import rag_manager
from app import metadata as meta_module
from app.prompts import (
    RETRIEVAL_ROUTER_PROMPT,
    DOCUMENT_GRADER_PROMPT,
    QUERY_REWRITER_PROMPT,
    GENERATE_PROMPT,
    RETRIEVAL_FAILED_PROMPT,
    ANALYZE_REPO_PROMPT,
)
from app.tools import list_repo_files, read_repo_file, search_repo_code, search_repo_rag


# ---------------------------------------------------------------------------
# Step 0: Retrieval Router
# ---------------------------------------------------------------------------
def make_retrieval_router_node(grader_llm):
    """
    Returns a node that uses the grader LLM to decide whether the user's
    question requires semantic retrieval from the vector store.
    """

    def retrieval_router_node(state: AgentState):
        """
        Decides: does this question NEED vector DB retrieval, or is parametric
        knowledge (or a direct tool call) sufficient?
        """
        question = state.get("planner_query") or state.get("question", "")

        # If there's no indexed repo, skip retrieval regardless
        if not rag_manager.is_indexed():
            return {
                "retrieval_needed": False,
                "loop_count":       0,
                "documents":        [],
                "generation":       "",
            }

        messages = [
            {"role": "system", "content": RETRIEVAL_ROUTER_PROMPT},
            {"role": "user",   "content": question},
        ]
        raw     = grader_llm.invoke(messages)
        content = raw.content.strip()

        retrieval_needed = True  # safe default
        try:
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            decision         = json.loads(content)
            retrieval_needed = bool(decision.get("retrieval_needed", True))
        except (json.JSONDecodeError, AttributeError):
            retrieval_needed = True

        return {
            "retrieval_needed": retrieval_needed,
            "loop_count":       0,
            "documents":        [],
            "generation":       "",
        }

    return retrieval_router_node


# ---------------------------------------------------------------------------
# Step 1: Retrieve
# ---------------------------------------------------------------------------
def make_retrieve_node():
    """Returns a retrieve node (no LLM needed — pure vector store lookup)."""

    def retrieve_node(state: AgentState):
        """
        Queries the vector store for the top-K chunks relevant to the current
        query. Uses planner_query for the initial retrieval; falls back to
        state['question'] on rewrites.
        """
        query      = state.get("planner_query") or state.get("question", "")
        loop_count = state.get("loop_count", 0)
        if loop_count > 0:
            query = state.get("question", query)

        docs = rag_manager.retrieve_chunks(query, k=5)
        return {"documents": docs}

    return retrieve_node


# ---------------------------------------------------------------------------
# Step 2 (CRAG): Grade Documents
# ---------------------------------------------------------------------------
def make_grade_documents_node(grader_llm):
    """
    Returns a node that evaluates each retrieved chunk with the grader LLM
    and discards irrelevant ones.
    """

    def grade_documents_node(state: AgentState):
        """
        Evaluates each retrieved chunk with the grader LLM.
        Irrelevant chunks are discarded.
        """
        question = state.get("planner_query") or state.get("question", "")
        if state.get("loop_count", 0) > 0:
            question = state.get("question", question)

        docs          = state.get("documents", [])
        relevant_docs = []

        for doc in docs:
            messages = [
                {"role": "system", "content": DOCUMENT_GRADER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Document:\n{doc.page_content[:600]}"
                    ),
                },
            ]
            raw     = grader_llm.invoke(messages)
            content = raw.content.strip()

            try:
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                score = json.loads(content)
                if score.get("relevant", "no").lower() == "yes":
                    relevant_docs.append(doc)
            except (json.JSONDecodeError, AttributeError):
                relevant_docs.append(doc)   # fail open

        # Lazy chunk summarisation — update metadata for newly seen chunks
        repo_path = rag_manager.get_current_repo_path()
        if repo_path and relevant_docs:
            current_meta = state.get("repo_metadata") or rag_manager.current_metadata
            updated_meta = meta_module.update_chunk_summaries(
                repo_path, relevant_docs, grader_llm, current_meta
            )
            rag_manager.set_metadata(updated_meta)
            return {"documents": relevant_docs, "repo_metadata": updated_meta}

        return {"documents": relevant_docs}

    return grade_documents_node


# ---------------------------------------------------------------------------
# Step 3 (CRAG): Rewrite Query
# ---------------------------------------------------------------------------
def make_rewrite_query_node(grader_llm):
    """Returns a node that rewrites the query for better retrieval on retry."""

    def rewrite_query_node(state: AgentState):
        """
        Uses the grader LLM to produce a semantically richer rewrite of the
        current question, then increments loop_count to track retries.
        """
        question   = state.get("question", "")
        if state.get("loop_count", 0) == 0 and state.get("planner_query"):
            question = state["planner_query"]

        loop_count = state.get("loop_count", 0) + 1

        messages = [
            {"role": "system", "content": QUERY_REWRITER_PROMPT},
            {"role": "user",   "content": f"Original question: {question}"},
        ]
        raw       = grader_llm.invoke(messages)
        rewritten = raw.content.strip().strip('"')

        return {"question": rewritten, "loop_count": loop_count}

    return rewrite_query_node


# ---------------------------------------------------------------------------
# Step 4: Generate
# ---------------------------------------------------------------------------
def make_generate_node(llm):
    """Returns a generation node that produces a grounded answer from verified chunks."""

    def generate_node(state: AgentState):
        """
        Formats the verified relevant chunks into a context string and asks the
        main LLM to generate a grounded answer. Injects the Planner's plan and
        the repo overview as additional context.
        """
        question = state.get("question", "")
        docs     = state.get("documents", [])
        plan     = state.get("plan", "")

        repo_meta = state.get("repo_metadata") or rag_manager.current_metadata
        overview  = repo_meta.get("overview", "")

        chunk_summaries = repo_meta.get("chunk_summaries", {})
        context_parts   = []
        for i, doc in enumerate(docs, 1):
            source     = doc.metadata.get("source", "unknown")
            cid        = meta_module.chunk_id(doc)
            summary_info = chunk_summaries.get(cid, {})
            summary    = summary_info.get("summary", "")
            header     = f"--- Chunk {i} | File: {source}"
            if summary:
                header += f" | Summary: {summary}"
            header += " ---"
            context_parts.append(f"{header}\n{doc.page_content}")

        context = "\n\n".join(context_parts) if context_parts else "No relevant context was retrieved."

        plan_section     = f"\nSEARCH PLAN:\n{plan}\n"     if plan     else ""
        overview_section = f"\nREPO OVERVIEW:\n{overview}\n" if overview else ""

        messages = [
            {
                "role":    "system",
                "content": GENERATE_PROMPT + plan_section + overview_section,
            },
            {
                "role":    "user",
                "content": (
                    f"VERIFIED RELEVANT CONTEXT:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ]
        response   = llm.invoke(messages)
        generation = response.content
        return {"generation": generation}

    return generate_node


# ---------------------------------------------------------------------------
# Step 5 (Self-RAG): Reflect — passthrough; routing logic lives in edges.py
# ---------------------------------------------------------------------------
def reflect_node(state: AgentState):
    """
    Self-RAG reflection step.
    This node does NOT change state — routing decisions are made by
    reflect_condition in edges.py which reads state directly.
    """
    return {}


# ---------------------------------------------------------------------------
# Finalize — push generation into messages
# ---------------------------------------------------------------------------
def make_finalize_node():
    """Returns a node that converts the generation string into an AIMessage."""

    def finalize_node(state: AgentState):
        """Converts the Self-CRAG 'generation' string into an AIMessage."""
        generation = state.get("generation", "I was unable to find a relevant answer.")
        return {"messages": [AIMessage(content=generation)]}

    return finalize_node


# ---------------------------------------------------------------------------
# Fallback — max retries exhausted
# ---------------------------------------------------------------------------
def make_fallback_node(llm):
    """Returns a fallback node for when MAX_RETRIES is exceeded."""

    def fallback_node(state: AgentState):
        """Tells the user what happened and suggests alternatives."""
        question = state.get("question", "your question")
        messages = [
            {"role": "system", "content": RETRIEVAL_FAILED_PROMPT},
            {"role": "user",   "content": f"Question that failed: {question}"},
        ]
        response = llm.invoke(messages)
        return {"messages": [AIMessage(content=response.content)]}

    return fallback_node


# ---------------------------------------------------------------------------
# Direct Analyze — tool-based analysis when RAG is skipped
# ---------------------------------------------------------------------------
def make_direct_analyze_node(llm, analyze_tools):
    """
    Returns a node that handles analysis queries that do NOT need RAG,
    or when no repo is indexed. Falls back to tool-based analysis.
    """

    def direct_analyze_node(state: AgentState):
        """
        Handles analysis queries that the router decided do NOT need RAG.
        The planner's tool_hints are surfaced in the prompt if available.
        """
        messages = state["messages"]
        last_msg = messages[-1] if messages else None

        if isinstance(last_msg, HumanMessage):
            user_query  = last_msg.content
            rag_context = rag_manager.query(user_query, k=5) if rag_manager.is_indexed() else "No repository indexed."

            hints    = state.get("tool_hints", [])
            hint_str = ""
            if hints:
                hint_str = f"\n\nPlanner suggests using these tools: {', '.join(hints)}"

            system_content = ANALYZE_REPO_PROMPT.format(rag_context=rag_context) + hint_str
            full_messages  = [{"role": "system", "content": system_content}] + messages
        else:
            full_messages = messages

        llm_with_tools = llm.bind_tools(analyze_tools)
        response       = llm_with_tools.invoke(full_messages)
        return {"messages": [response]}

    return direct_analyze_node
