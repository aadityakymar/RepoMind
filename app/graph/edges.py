"""
app/graph/edges.py — All conditional-edge routing functions.

Plain routing functions (no LLM dependency):
    route_intent, router_condition, grade_condition, analyze_tools_condition

Factory for the LLM-dependent routing function:
    make_reflect_condition(grader_llm) → reflect_condition
"""

import json

from app.state import AgentState
from app.rag import rag_manager
from app.config import MAX_RETRIES
from app.prompts import HALLUCINATION_GRADER_PROMPT, ANSWER_GRADER_PROMPT


# ---------------------------------------------------------------------------
# Intent router (after classifier node)
# ---------------------------------------------------------------------------
def route_intent(state: AgentState) -> str:
    """Routes after classifier based on detected intent."""
    intent = state.get("intent", "chat")
    if intent == "clone_repo":
        return "clone_repo"
    elif intent == "analyze_repo":
        return "analyze_repo"   # → planner node
    elif intent == "index_path":
        return "index_path"     # → index_path_node
    return "chat"


# ---------------------------------------------------------------------------
# Retrieval router condition (after retrieval_router_node)
# ---------------------------------------------------------------------------
def router_condition(state: AgentState) -> str:
    """
    After retrieval_router_node: branch to retrieve or skip straight to
    direct analysis (no RAG needed).
    """
    if state.get("retrieval_needed", True):
        return "retrieve"
    return "direct_analyze"


# ---------------------------------------------------------------------------
# Grade condition (after grade_documents_node)
# ---------------------------------------------------------------------------
def grade_condition(state: AgentState) -> str:
    """
    After grade_documents_node:
      - Some relevant docs → proceed to generate
      - No relevant docs + retries left → rewrite query
      - No relevant docs + retries exhausted → fallback
    """
    docs       = state.get("documents", [])
    loop_count = state.get("loop_count", 0)

    if docs:
        return "generate"
    if loop_count >= MAX_RETRIES:
        return "fallback"
    return "rewrite"


# ---------------------------------------------------------------------------
# Analyze-tools condition (inside the direct_analyze loop)
# ---------------------------------------------------------------------------
def analyze_tools_condition(state: AgentState) -> str:
    """Used in the direct_analyze branch: check if the LLM wants to call tools."""
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    if last_msg is not None and getattr(last_msg, "tool_calls", None):
        return "use_tools"
    return "done"


# ---------------------------------------------------------------------------
# Clone proceed condition (after clone_repo_node)
# ---------------------------------------------------------------------------
def clone_proceed_condition(state: AgentState) -> str:
    """
    After clone_repo_node:
      - If the node made a tool call (clone confirmed) → proceed to tool_call
      - If the node returned a plain AIMessage (clone cancelled) → END
    """
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    if last_msg is not None and getattr(last_msg, "tool_calls", None):
        return "proceed"
    return "cancelled"


# ---------------------------------------------------------------------------
# Reflect condition factory (Self-RAG — requires grader_llm)
# ---------------------------------------------------------------------------
def make_reflect_condition(grader_llm):
    """
    Returns a reflect_condition function closed over grader_llm.
    Performs hallucination and usefulness checks, then routes to
    'finalize' or 'rewrite'.
    """
    def reflect_condition(state: AgentState) -> str:
        """
        After reflect_node (Self-RAG):
          1. Check for hallucination
          2. Check for usefulness
          Route accordingly.
        """
        question   = state.get("question", "")
        generation = state.get("generation", "")
        docs       = state.get("documents", [])
        loop_count = state.get("loop_count", 0)

        # Guard: if too many loops, just finalize whatever we have
        if loop_count >= MAX_RETRIES:
            return "finalize"

        # --- Hallucination Check ---
        doc_texts = "\n\n".join(
            f"[{doc.metadata.get('source','?')}]: {doc.page_content[:400]}"
            for doc in docs
        )
        hal_messages = [
            {"role": "system", "content": HALLUCINATION_GRADER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Source Documents:\n{doc_texts or 'No documents.'}\n\n"
                    f"Generated Answer:\n{generation}"
                ),
            },
        ]
        hal_raw     = grader_llm.invoke(hal_messages)
        hal_content = hal_raw.content.strip()

        grounded = True  # safe default
        try:
            if hal_content.startswith("```"):
                hal_content = hal_content.split("```")[1]
                if hal_content.startswith("json"):
                    hal_content = hal_content[4:]
            hal_score = json.loads(hal_content)
            grounded  = hal_score.get("grounded", "yes").lower() == "yes"
        except (json.JSONDecodeError, AttributeError):
            grounded = True

        if not grounded:
            return "rewrite"

        # --- Usefulness Check ---
        ans_messages = [
            {"role": "system", "content": ANSWER_GRADER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Answer: {generation}"
                ),
            },
        ]
        ans_raw     = grader_llm.invoke(ans_messages)
        ans_content = ans_raw.content.strip()

        useful = True
        try:
            if ans_content.startswith("```"):
                ans_content = ans_content.split("```")[1]
                if ans_content.startswith("json"):
                    ans_content = ans_content[4:]
            ans_score = json.loads(ans_content)
            useful    = ans_score.get("useful", "yes").lower() == "yes"
        except (json.JSONDecodeError, AttributeError):
            useful = True

        return "finalize" if useful else "rewrite"

    return reflect_condition
