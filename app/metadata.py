"""
metadata.py — Persistent Repository Metadata

Generates, saves, and loads a JSON metadata file for each cloned repository at:
    repos/<repo_name>/.meta/repo_metadata.json

Structure:
{
    "repo_name": str,
    "overview": str,           # 3–5 sentence description of the repo
    "main_language": str,      # dominant language detected
    "key_files": [str, ...],   # structurally important files
    "chunk_summaries": {       # populated lazily on retrieval
        "<chunk_id>": {
            "file": str,
            "summary": str
        }
    }
}
"""

import os
import json
import hashlib
from typing import Optional

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
META_DIR = ".meta"
META_FILENAME = "repo_metadata.json"


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------
def _meta_path(repo_path: str) -> str:
    """Returns the absolute path to the metadata JSON file for a given repo."""
    return os.path.join(repo_path, META_DIR, META_FILENAME)


# ---------------------------------------------------------------------------
# Chunk ID — stable hash so we can look up summaries across sessions
# ---------------------------------------------------------------------------
def chunk_id(doc: Document) -> str:
    """
    Produces a stable, short ID for a Document chunk based on its source file
    and the first 200 characters of its content.
    """
    payload = f"{doc.metadata.get('source', '')}::{doc.page_content[:200]}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------
def generate_repo_metadata(repo_path: str, documents: list[Document], llm) -> dict:
    """
    Uses the LLM to generate a high-level overview of the repository and
    identify structurally important files.

    Args:
        repo_path:  Absolute path to the cloned repository.
        documents:  List of Document objects loaded from the repo (pre-split).
        llm:        A LangChain LLM instance (used for one overview call).

    Returns:
        A metadata dict ready to be saved with save_metadata().
    """
    from prompts import REPO_OVERVIEW_PROMPT  # local import to avoid circular deps

    repo_name = os.path.basename(repo_path)

    # Build a compact file manifest for the LLM (file names + first 120 chars each)
    file_snippets = []
    for doc in documents[:40]:  # cap at 40 files to stay within token budget
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:120].replace("\n", " ").strip()
        file_snippets.append(f"  [{source}]: {preview}")

    manifest = "\n".join(file_snippets) if file_snippets else "  (no files found)"

    # Single LLM call to generate overview + key files
    messages = [
        {"role": "system", "content": REPO_OVERVIEW_PROMPT},
        {
            "role": "user",
            "content": (
                f"Repository name: {repo_name}\n\n"
                f"File manifest (file path → content preview):\n{manifest}"
            ),
        },
    ]

    raw = llm.invoke(messages)
    content = raw.content.strip()

    # Parse JSON response
    overview = ""
    main_language = "unknown"
    key_files: list[str] = []

    try:
        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        overview = parsed.get("overview", "")
        main_language = parsed.get("main_language", "unknown")
        key_files = parsed.get("key_files", [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"[Metadata] Warning — could not parse overview JSON: {e}")
        overview = content  # fall back to raw text

    metadata = {
        "repo_name": repo_name,
        "overview": overview,
        "main_language": main_language,
        "key_files": key_files,
        "chunk_summaries": {},  # populated lazily
    }

    print(f"[Metadata] Overview generated for '{repo_name}'.")
    return metadata


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------
def save_metadata(repo_path: str, metadata: dict) -> None:
    """
    Persists the metadata dict as JSON to repos/<repo_name>/.meta/repo_metadata.json.
    Creates the .meta directory if it doesn't exist.
    """
    meta_dir = os.path.join(repo_path, META_DIR)
    os.makedirs(meta_dir, exist_ok=True)
    path = _meta_path(repo_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Metadata] Saved to: {path}")


def load_metadata(repo_path: str) -> dict:
    """
    Loads and returns the metadata dict for a repository.
    Returns an empty dict if the metadata file does not exist yet.
    """
    path = _meta_path(repo_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[Metadata] Warning — could not load metadata: {e}")
        return {}


# ---------------------------------------------------------------------------
# Lazy chunk summarisation
# ---------------------------------------------------------------------------
def summarise_chunk(doc: Document, grader_llm) -> str:
    """
    Generates a one-line summary of a single Document chunk using the grader LLM.
    This is called lazily (only when the chunk is actually retrieved and used).

    Args:
        doc:        The Document chunk to summarise.
        grader_llm: A fast, cheap LLM instance.

    Returns:
        A one-line string summary.
    """
    from prompts import CHUNK_SUMMARY_PROMPT  # local import to avoid circular deps

    messages = [
        {"role": "system", "content": CHUNK_SUMMARY_PROMPT},
        {
            "role": "user",
            "content": (
                f"File: {doc.metadata.get('source', 'unknown')}\n\n"
                f"Code:\n{doc.page_content[:600]}"
            ),
        },
    ]
    raw = grader_llm.invoke(messages)
    return raw.content.strip().splitlines()[0]  # take first line only


def update_chunk_summaries(
    repo_path: str,
    docs: list[Document],
    grader_llm,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Lazily generates and persists summaries for any chunks that haven't been
    summarised yet. Modifies and returns the metadata dict in-place.

    Args:
        repo_path:  Absolute path to the repo (for saving updates).
        docs:       List of retrieved Document chunks.
        grader_llm: Fast LLM instance for summarisation.
        metadata:   Existing metadata dict; loads from disk if None.

    Returns:
        Updated metadata dict.
    """
    if metadata is None:
        metadata = load_metadata(repo_path)

    chunk_summaries: dict = metadata.setdefault("chunk_summaries", {})
    updated = False

    for doc in docs:
        cid = chunk_id(doc)
        if cid not in chunk_summaries:
            summary = summarise_chunk(doc, grader_llm)
            chunk_summaries[cid] = {
                "file": doc.metadata.get("source", "unknown"),
                "summary": summary,
            }
            updated = True
            print(f"[Metadata] Chunk summary: [{cid}] {summary[:60]}")

    if updated:
        save_metadata(repo_path, metadata)

    return metadata
