import os
from typing import Optional

# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Supported file extensions to index into the vector store
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".go", ".rs", ".rb", ".php", ".cs", ".swift", ".kt",
    ".md", ".txt", ".rst", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    ".html", ".css", ".sh", ".bash", ".env.example",
    ".ipynb",  # Jupyter notebooks
    ".r", ".R",  # R scripts
    ".sql",  # SQL files
}

# Max file size to index (skip very large files like lock files)
MAX_FILE_SIZE_BYTES = 500_000  # 500 KB


class VectorStoreManager:
    """
    Manages an in-memory Chroma vector store for a single repository session.
    Lifecycle:
      - index_repository()  â†’ called after a repo is cloned
      - retrieve_chunks()   â†’ called by Self-CRAG retrieve_node; returns raw Documents
      - query()             â†’ legacy helper; returns a formatted string for simple use
      - clear()             â†’ called when the session ends (like freeing RAM)

    Additional helpers for Planner + Metadata integration:
      - get_current_repo_path() â†’ returns the absolute path to the indexed repo
      - current_metadata        â†’ cached metadata dict for the active repo
    """

    def __init__(self):
        self._vectorstore: Optional[Chroma] = None
        self._repo_name: Optional[str] = None
        self._repo_path: Optional[str] = None      # NEW: track repo path
        self._metadata: dict = {}                  # NEW: cached metadata

        # Using a free, local embedding model â€” no API key required
        # Downloads ~90MB once and caches it locally
        print("[*] Loading embedding model (MiniLM-L6-v2)... ", end="", flush=True)
        self._embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("done.")

    def is_indexed(self) -> bool:
        """Returns True if a repository is currently indexed."""
        return self._vectorstore is not None

    @property
    def current_repo(self) -> Optional[str]:
        return self._repo_name

    # NEW: expose repo path so graph.py / metadata.py can reference it
    def get_current_repo_path(self) -> Optional[str]:
        """Returns the absolute path to the currently indexed repository, or None."""
        return self._repo_path

    # NEW: expose cached metadata dict
    @property
    def current_metadata(self) -> dict:
        """Returns the loaded metadata dict for the active repo (may be empty)."""
        return self._metadata

    # NEW: allow graph.py to update the cached metadata after generating it
    def set_metadata(self, metadata: dict) -> None:
        """Caches the metadata dict for the active repo."""
        self._metadata = metadata

    # ---------------------------------------------------------------------------
    # index_path â€” index any file or folder within the repos directory
    # ---------------------------------------------------------------------------
    def index_path(self, target_path: str) -> tuple[str, list[Document]]:
        """
        Indexes a specific file or directory into the vector store.
        Unlike index_repository(), this does NOT require a full repo clone â€”
        it can target any file or subfolder inside the repos/ directory.

        The 'indexed name' is set to the basename of the target path so the
        rest of the pipeline (planner, generate, etc.) can reference it.

        Args:
            target_path (str): Absolute path to a file or directory to index.

        Returns:
            A tuple of:
              - status string
              - list of raw Document objects (pre-split, for metadata generation)
        """
        if not os.path.exists(target_path):
            return f"[!] Path does not exist: {target_path}", []

        # Clear any existing index
        if self._vectorstore is not None:
            self.clear()

        index_name = os.path.basename(target_path.rstrip("/\\"))
        self._repo_name = index_name
        self._repo_path = target_path

        documents: list[Document] = []
        skipped = 0

        # ---- Single file ----
        if os.path.isfile(target_path):
            ext = os.path.splitext(target_path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                return (
                    f"[!] File extension '{ext}' is not in the supported list. "
                    f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                ), []
            try:
                if os.path.getsize(target_path) > MAX_FILE_SIZE_BYTES:
                    return f"[!] File is too large (> {MAX_FILE_SIZE_BYTES // 1024} KB): {target_path}", []
                with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if not content.strip():
                    return f"[!] File is empty: {target_path}", []
                doc = Document(
                    page_content=content,
                    metadata={"source": os.path.basename(target_path), "repo": index_name}
                )
                documents.append(doc)
                print(f"\n[*] Indexing single file '{index_name}' into vector store...")
            except Exception as e:
                return f"[!] Error reading file: {e}", []

        # ---- Directory ----
        else:
            print(f"\n[*] Indexing folder '{index_name}' into vector store...")
            for root, dirs, files in os.walk(target_path):
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith(".")
                    and d not in ("node_modules", "__pycache__", "venv", ".venv", "dist", "build")
                ]
                for filename in files:
                    file_path = os.path.join(root, filename)
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        continue
                    try:
                        if os.path.getsize(file_path) > MAX_FILE_SIZE_BYTES:
                            skipped += 1
                            continue
                    except OSError:
                        continue
                    try:
                        rel_path = os.path.relpath(file_path, target_path)
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        if not content.strip():
                            continue
                        doc = Document(
                            page_content=content,
                            metadata={"source": rel_path, "repo": index_name}
                        )
                        documents.append(doc)
                    except Exception:
                        skipped += 1
                        continue

        if not documents:
            return f"[!] No indexable files found at: {target_path}", []

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            collection_name=f"path_{index_name}",
        )

        status = (
            f"[OK] Indexed {len(chunks)} chunks from {len(documents)} file(s) "
            f"({skipped} skipped) for '{index_name}'."
        )
        return status, documents

    def index_repository(self, repo_path: str) -> tuple[str, list[Document]]:
        """
        Walks the repository, loads all supported text files, splits them into
        chunks and stores them in an in-memory Chroma collection.

        Args:
            repo_path (str): Absolute path to the cloned repository.

        Returns:
            A tuple of:
              - status string describing how many chunks were indexed
              - list of raw (pre-split) Document objects (used by metadata.py)
        """
        # Clear any existing index first
        if self._vectorstore is not None:
            self.clear()

        repo_name = os.path.basename(repo_path)
        self._repo_name = repo_name
        self._repo_path = repo_path  # NEW

        documents = []
        skipped = 0

        print(f"\n[*] Indexing '{repo_name}' into vector store...")

        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories (e.g. .git, .venv, node_modules)
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".")
                and d not in ("node_modules", "__pycache__", "venv", ".venv", "dist", "build")
            ]

            for filename in files:
                file_path = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()

                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                # Skip very large files
                try:
                    if os.path.getsize(file_path) > MAX_FILE_SIZE_BYTES:
                        skipped += 1
                        continue
                except OSError:
                    continue

                try:
                    rel_path = os.path.relpath(file_path, repo_path)
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if not content.strip():
                        continue
                    doc = Document(
                        page_content=content,
                        metadata={"source": rel_path, "repo": repo_name}
                    )
                    documents.append(doc)
                except Exception:
                    skipped += 1
                    continue

        if not documents:
            status = f"[!] No indexable files found in '{repo_name}'."
            return status, []

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        # Build the in-memory Chroma vector store
        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            collection_name=f"repo_{repo_name}"
        )

        status = (
            f"[OK] Indexed {len(chunks)} chunks from {len(documents)} files "
            f"({skipped} skipped) in '{repo_name}'."
        )
        # Return both status and the pre-split documents for metadata generation
        return status, documents

    # -----------------------------------------------------------------------
    # NEW: Self-CRAG retrieval â€” returns raw Document objects for grading
    # -----------------------------------------------------------------------
    def retrieve_chunks(self, question: str, k: int = 5) -> list[Document]:
        """
        Performs a semantic similarity search and returns the top-k raw Document
        chunks. These are NOT pre-formatted so that the CRAG document grader can
        inspect each chunk individually and decide whether it is relevant.

        Args:
            question (str): The user's question or (rewritten) query.
            k (int): Number of chunks to retrieve.

        Returns:
            A list of Document objects, or an empty list if not indexed / no results.
        """
        if self._vectorstore is None:
            return []

        try:
            return self._vectorstore.similarity_search(question, k=k)
        except Exception as e:
            print(f"[!] retrieve_chunks error: {e}")
            return []

    # -----------------------------------------------------------------------
    # Legacy helper â€” kept for tools.py / backward compatibility
    # -----------------------------------------------------------------------
    def query(self, question: str, k: int = 5) -> str:
        """
        Performs a semantic similarity search and returns the top-k chunks
        as a formatted context string to be injected into the LLM prompt.

        Args:
            question (str): The user's question or query.
            k (int): Number of chunks to retrieve.

        Returns:
            A formatted string of relevant code/text chunks with file citations.
        """
        if self._vectorstore is None:
            return "No repository is currently indexed. Please clone a repository first."

        try:
            results = self._vectorstore.similarity_search(question, k=k)
        except Exception as e:
            return f"Error querying vector store: {str(e)}"

        if not results:
            return "No relevant code found for your query."

        parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"--- Chunk {i} | File: {source} ---\n{doc.page_content}")

        return "\n\n".join(parts)

    def clear(self):
        """
        Clears the current vector store from memory (simulates RAM free).
        Called when the chat session ends.
        """
        if self._vectorstore is not None:
            try:
                self._vectorstore.delete_collection()
            except Exception:
                pass
            self._vectorstore = None
            self._repo_name = None
            self._repo_path = None   # NEW
            self._metadata = {}      # NEW
            print("\n[*] Vector store cleared. Session ended.")


# ---------------------------------------------------------------------------
# Singleton instance â€” imported and shared by tools.py and graph.py
# ---------------------------------------------------------------------------
rag_manager = VectorStoreManager()

