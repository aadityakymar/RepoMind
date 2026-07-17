"""
app/graph/nodes/clone.py — Clone-repo pipeline nodes.

Nodes:
    make_clone_repo_node   → forces clone_github_repo tool call
    make_index_repo_node   → parses clone result, indexes repo, generates metadata
    make_chat_update_node  → post-clone friendly summary
"""

import os
import re

from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langgraph.types import interrupt

from app.state import AgentState
from app.rag import rag_manager
from app import metadata as meta_module
from app.prompts import CLONE_REPO_PROMPT, CHAT_UPDATE_PROMPT
from app.tools import clone_github_repo


def make_clone_repo_node(llm):
    """Returns a node that asks for human confirmation before cloning."""

    def clone_repo_node(state: AgentState):
        """
        Human-in-the-loop: pauses and asks the user to confirm before cloning.
        If confirmed, forces the clone_github_repo tool call.
        If cancelled, returns an AIMessage and the graph routes to END.
        """
        # Extract GitHub URL from the latest human message for a clear prompt
        url_hint = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                urls = re.findall(
                    r'https?://github\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?',
                    msg.content,
                )
                if urls:
                    url_hint = f"\n  \033[96m→ {urls[0]}\033[0m"
                break

        # ── Human-in-the-loop pause ──────────────────────────────────────────
        answer = interrupt(
            f"\n  Clone this repository?{url_hint}"
            f"\n  Type \033[92myes\033[0m to proceed or \033[91mno\033[0m to cancel."
        )

        if str(answer).strip().lower() not in ("yes", "y"):
            return {
                "messages": [
                    AIMessage(content="Got it — clone cancelled. Let me know if you'd like to try a different repository.")
                ]
            }

        # ── Proceed with clone ───────────────────────────────────────────────
        messages = [{"role": "system", "content": CLONE_REPO_PROMPT}] + state["messages"]
        llm_with_tool = llm.bind_tools([clone_github_repo], tool_choice="clone_github_repo")
        response = llm_with_tool.invoke(messages)
        return {"messages": [response]}

    return clone_repo_node


def make_index_repo_node(llm):
    """
    Returns a node that reads the tool result from the clone, extracts
    repo_name + repo_path, triggers RAG indexing, generates + saves persistent
    repo metadata, and updates state with the active repo name and metadata.
    """

    def index_repo_node(state: AgentState):
        messages = state["messages"]

        # Find the most recent ToolMessage (clone result)
        tool_output = ""
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_output = msg.content
                break

        # Parse structured output: "REPO_NAME:<name>|REPO_PATH:<path>|..."
        repo_name = ""
        repo_path = ""

        if "REPO_NAME:" in tool_output:
            for part in tool_output.split("|"):
                if part.startswith("REPO_NAME:"):
                    repo_name = part[len("REPO_NAME:"):]
                elif part.startswith("REPO_PATH:"):
                    repo_path = part[len("REPO_PATH:"):]

        loaded_metadata = {}

        # Trigger indexing only on a fresh successful clone (repo_path will be set)
        if repo_path and os.path.exists(repo_path):
            _status, documents = rag_manager.index_repository(repo_path)

            existing_meta = meta_module.load_metadata(repo_path)
            if existing_meta:
                loaded_metadata = existing_meta
            else:
                loaded_metadata = meta_module.generate_repo_metadata(
                    repo_path, documents, llm
                )
                meta_module.save_metadata(repo_path, loaded_metadata)

            rag_manager.set_metadata(loaded_metadata)

        elif repo_name and not repo_path:
            # Already exists case — re-index from the repos folder
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Navigate up from app/graph/nodes/ to the project root, then into repos/
            project_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
            existing_path = os.path.join(project_root, "repos", repo_name)
            if os.path.exists(existing_path) and not rag_manager.is_indexed():
                _status, _docs = rag_manager.index_repository(existing_path)
                loaded_metadata = meta_module.load_metadata(existing_path)
                rag_manager.set_metadata(loaded_metadata)

        return {
            "repo_name":     repo_name,
            "repo_metadata": loaded_metadata,
        }

    return index_repo_node


def make_chat_update_node(llm):
    """Returns a node that reads the clone + indexing result and gives a friendly summary."""

    def chat_update_node(state: AgentState):
        """Reads the clone + indexing result and gives a friendly summary."""
        messages = [{"role": "system", "content": CHAT_UPDATE_PROMPT}] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    return chat_update_node
