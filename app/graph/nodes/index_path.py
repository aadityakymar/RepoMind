"""
app/graph/nodes/index_path.py — Index-path pipeline nodes.

Nodes:
    make_index_path_node        → forces index_path_for_rag tool call
    make_index_path_update_node → parses tool result, updates metadata, confirms
"""

import os

from langchain_core.messages import ToolMessage

from app.state import AgentState
from app.rag import rag_manager
from app import metadata as meta_module
from app.prompts import INDEX_PATH_PROMPT, INDEX_PATH_UPDATE_PROMPT
from app.tools import index_path_for_rag


def make_index_path_node(llm):
    """
    Returns a node that extracts the target file/folder path from the user's
    message and forces a call to the index_path_for_rag tool.
    """

    def index_path_node(state: AgentState):
        """Extracts the target path and forces the index_path_for_rag tool call."""
        messages = [{"role": "system", "content": INDEX_PATH_PROMPT}] + state["messages"]
        llm_with_tool = llm.bind_tools([index_path_for_rag], tool_choice="index_path_for_rag")
        response = llm_with_tool.invoke(messages)
        return {"messages": [response]}

    return index_path_node


def make_index_path_update_node(llm):
    """
    Returns a node that reads the index_path_for_rag tool result, extracts
    INDEX_NAME + INDEX_PATH, loads or generates metadata for the newly indexed
    path, and emits a friendly confirmation message.
    """

    def index_path_update_node(state: AgentState):
        messages = state["messages"]

        tool_output = ""
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_output = msg.content
                break

        index_name = ""
        index_path = ""
        loaded_metadata = state.get("repo_metadata") or {}

        if "INDEX_NAME:" in tool_output:
            for part in tool_output.split("|"):
                if part.startswith("INDEX_NAME:"):
                    index_name = part[len("INDEX_NAME:"):]
                elif part.startswith("INDEX_PATH:"):
                    index_path = part[len("INDEX_PATH:"):]

        # For directories: load existing metadata or build a lightweight one.
        # For single files: skip metadata (no meaningful directory-level overview).
        if index_path and os.path.isdir(index_path) and not tool_output.startswith("[!"):
            existing_meta = meta_module.load_metadata(index_path)
            if existing_meta:
                loaded_metadata = existing_meta
            else:
                loaded_metadata = {
                    "repo_name":       index_name,
                    "overview":        f"RAG index focused on folder: {index_path}",
                    "main_language":   "unknown",
                    "key_files":       [],
                    "chunk_summaries": {},
                }
                meta_module.save_metadata(index_path, loaded_metadata)
            rag_manager.set_metadata(loaded_metadata)
        elif index_path and os.path.isfile(index_path) and not tool_output.startswith("[!"):
            loaded_metadata = {
                "repo_name":       index_name,
                "overview":        f"RAG index focused on file: {index_path}",
                "main_language":   "unknown",
                "key_files":       [os.path.basename(index_path)],
                "chunk_summaries": {},
            }
            rag_manager.set_metadata(loaded_metadata)

        # Emit a friendly summary
        full_messages = [{"role": "system", "content": INDEX_PATH_UPDATE_PROMPT}] + messages
        response = llm.invoke(full_messages)
        return {
            "messages":      [response],
            "repo_name":     index_name or state.get("repo_name", ""),
            "repo_metadata": loaded_metadata,
        }

    return index_path_update_node
