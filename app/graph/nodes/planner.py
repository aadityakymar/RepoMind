"""app/graph/nodes/planner.py — Planner node."""

import json

from langchain_core.messages import HumanMessage

from app.state import AgentState
from app.rag import rag_manager
from app.prompts import PLANNER_PROMPT


def make_planner_node(llm):
    """
    Returns a planner node that analyses the user's question + repo metadata to
    produce a step-by-step plan, enriched vector-search query, and tool hints.
    """

    def planner_node(state: AgentState):
        """
        Planner LLM node.

        Takes the raw user question and the repo's high-level metadata (overview,
        main language, key files) and produces a structured plan:
          - A step-by-step reasoning string explaining what to look for and why
          - A reformulated, vector-search-optimised query string
          - Optional tool hints (advisory — the router still makes the final call)

        The planner_query replaces 'question' in all downstream retrieval nodes.
        The plan is injected as context into generate_node for better answers.
        """
        # Extract the latest human question
        question = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                question = msg.content
                break

        # Build repo context from metadata (falls back gracefully if not yet available)
        repo_meta      = state.get("repo_metadata") or rag_manager.current_metadata
        overview       = repo_meta.get("overview", "No overview available.")
        main_language  = repo_meta.get("main_language", "unknown")
        key_files      = repo_meta.get("key_files", [])
        key_files_str  = ", ".join(key_files) if key_files else "unknown"

        repo_context = (
            f"Repository overview: {overview}\n"
            f"Main language: {main_language}\n"
            f"Key files: {key_files_str}"
        )

        messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"USER QUESTION: {question}\n\n"
                    f"REPO OVERVIEW:\n{repo_context}"
                ),
            },
        ]

        raw     = llm.invoke(messages)
        content = raw.content.strip()

        # Parse JSON response
        plan         = ""
        planner_query = question   # safe fallback to original question
        tool_hints: list = []

        try:
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed        = json.loads(content)
            plan          = parsed.get("plan", "")
            planner_query = parsed.get("reformulated_query", question)
            tool_hints    = parsed.get("tool_hints", [])
        except (json.JSONDecodeError, AttributeError):
            plan          = content   # store raw text as plan
            planner_query = question  # fall back to original

        return {
            "question":      question,       # preserve original question for traceability
            "plan":          plan,
            "planner_query": planner_query,
            "tool_hints":    tool_hints,
        }

    return planner_node
