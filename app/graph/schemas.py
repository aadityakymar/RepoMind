"""
app/graph/schemas.py — Pydantic models for structured LLM outputs.

Extracted from graph.py so that nodes and edges can import them
without depending on the full graph-construction module.
"""

from typing import Literal
from pydantic import BaseModel, Field


class Intent(BaseModel):
    intent: Literal["chat", "clone_repo", "analyze_repo", "index_path"] = Field(
        description=(
            "Classify the user's intent:\n"
            "- 'clone_repo': user wants to clone a GitHub repository (mentions a GitHub URL or 'clone').\n"
            "- 'analyze_repo': user asks about files, code, structure, or wants to read/search/explain a cloned repo.\n"
            "- 'index_path': user explicitly asks to index / load / use RAG on a specific file or folder "
            "  (e.g. 'use RAG for the src folder', 'index notebooks/eda.ipynb', 'load utils.py into RAG').\n"
            "- 'chat': anything else — general conversation, questions, greetings."
        )
    )
