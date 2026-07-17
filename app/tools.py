import os
import subprocess
from langchain_core.tools import tool

# Import the RAG manager singleton for semantic search
from app.rag import rag_manager


# ---------------------------------------------------------------------------
# Shared helper: locate a folder or file inside the repos/ directory
# ---------------------------------------------------------------------------
def find_in_repos(
    name: str,
    *,
    find_file: bool = False,
    base_dir: str = None,
) -> str | None:
    """
    Searches the ``repos/`` directory (the single canonical location for all
    cloned repositories) for a folder or file whose name matches *name*.

    Because all repos are always cloned into ``repos/``, we never need to
    walk the entire project tree â€” only that one subdirectory.

    Args:
        name       : The folder name (e.g. a repo name) or file name /
                     relative sub-path (e.g. ``'my-repo/src'``) to locate.
        find_file  : When ``True``, also match against *files* inside repos/.
                     When ``False`` (default), only folder names are matched.
        base_dir   : Absolute path of the project root. Defaults to the
                     directory that contains this file.

    Returns:
        The absolute path of the first match, or ``None`` if nothing is found.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    repos_dir = os.path.join(base_dir, "repos")
    if not os.path.isdir(repos_dir):
        return None

    # Normalise separators so callers can pass 'my-repo/src' on any OS
    name_normalised = name.replace("/", os.sep).replace("\\", os.sep)

    for root, dirs, files in os.walk(repos_dir):
        # Never descend into .git internals â€” they're large and irrelevant
        dirs[:] = [d for d in dirs if d != ".git"]

        # Match folders
        for d in dirs:
            candidate = os.path.join(root, d)
            if d == name or candidate.endswith(os.sep + name_normalised):
                return candidate

        # Optionally match files
        if find_file:
            for f in files:
                candidate = os.path.join(root, f)
                if f == name or candidate.endswith(os.sep + name_normalised):
                    return candidate

    return None


# ---------------------------------------------------------------------------
# Tool 1: Clone a GitHub Repository
# ---------------------------------------------------------------------------
@tool
def clone_github_repo(repo_url: str) -> str:
    """
    Clones a GitHub repository from the provided URL into the repos/ directory.

    All repositories are always cloned into the ``repos/`` folder that sits
    next to this file.  The directory is created automatically if it does not
    yet exist.

    Args:
        repo_url (str): The full HTTPS URL of the GitHub repository to clone
                        (e.g. ``https://github.com/user/my-repo.git``).
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        repos_dir = os.path.join(base_dir, "repos")
        os.makedirs(repos_dir, exist_ok=True)

        # Extract clean repository name from the URL
        # e.g. https://github.com/user/repo-name.git â†’ repo-name
        repo_name = repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        target_path = os.path.join(repos_dir, repo_name)

        if os.path.exists(target_path):
            return (
                f"REPO_NAME:{repo_name}|"
                f"Error: A repository named '{repo_name}' already exists at {target_path}. "
                f"You can ask questions about it directly."
            )

        subprocess.run(
            ["git", "clone", repo_url, target_path],
            capture_output=True,
            text=True,
            check=True,
        )

        # Return a structured string so the graph can parse out repo_name and path
        return (
            f"REPO_NAME:{repo_name}|"
            f"REPO_PATH:{target_path}|"
            f"Success! Repository '{repo_name}' cloned into: {target_path}"
        )

    except subprocess.CalledProcessError as e:
        return f"Failed to clone the repository. Git Error: {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred while trying to clone the repository: {str(e)}"


# ---------------------------------------------------------------------------
# Helper: Find the repository path inside repos/
# ---------------------------------------------------------------------------
def get_repo_path(base_dir: str, repo_name: str) -> str | None:
    """
    Returns the absolute path of *repo_name* inside the ``repos/`` directory,
    or ``None`` if it cannot be found.

    Delegates to :func:`find_in_repos` so that all filesystem searching is
    consolidated in one place.
    """
    return find_in_repos(repo_name, base_dir=base_dir)

# ---------------------------------------------------------------------------
# Tool 2: List Repository Files
# ---------------------------------------------------------------------------
@tool
def list_repo_files(repo_name: str, sub_path: str = "") -> str:
    """
    Lists all files inside a cloned repository (excluding .git internals).

    Args:
        repo_name (str): The name of the cloned repository folder.
        sub_path (str, optional): A sub-directory to list. Defaults to the root.
    """
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = get_repo_path(base_dir, repo_name)
    
    if not repo_dir:
        return f"Error: Could not find repository '{repo_name}' anywhere in the codebase."
        
    target_dir = os.path.join(repo_dir, sub_path)

    if not os.path.exists(target_dir):
        return f"Error: Path '{target_dir}' does not exist inside repository '{repo_name}'."

    try:
        files = []
        for root, dirs, filenames in os.walk(target_dir):
            dirs[:] = [d for d in dirs if d != ".git"]
            for filename in filenames:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, repo_dir)
                files.append(rel_path)

        if not files:
            label = sub_path if sub_path else "root"
            return f"The directory '{label}' is empty or contains no files."

        return "\n".join(sorted(files))
    except Exception as e:
        return f"Error listing files: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 3: Read a File from the Repository
# ---------------------------------------------------------------------------
@tool
def read_repo_file(repo_name: str, file_path: str) -> str:
    """
    Reads and returns the full contents of a specific file within a cloned repository.

    Args:
        repo_name (str): The name of the cloned repository.
        file_path (str): Relative path to the file inside the repo (e.g. 'src/main.py').
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = get_repo_path(base_dir, repo_name)
    
    if not repo_dir:
        return f"Error: Could not find repository '{repo_name}' anywhere in the codebase."
        
    target_file = os.path.join(repo_dir, file_path)

    if not os.path.exists(target_file):
        return f"Error: File '{file_path}' does not exist in repository '{repo_name}'."

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()
        # Truncate very large files to avoid overwhelming the context window
        if len(content) > 8000:
            return content[:8000] + f"\n\n... [File truncated â€” showing first 8000 chars of {len(content)} total]"
        return content
    except UnicodeDecodeError:
        return f"Error: '{file_path}' appears to be a binary file or uses an unsupported encoding."
    except Exception as e:
        return f"Error reading file: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 4: Keyword Search Across the Repository
# ---------------------------------------------------------------------------
@tool
def search_repo_code(repo_name: str, query: str) -> str:
    """
    Performs a keyword/text search across all files in a cloned repository.

    Args:
        repo_name (str): The name of the cloned repository.
        query (str): The exact text string to search for.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = get_repo_path(base_dir, repo_name)
    
    if not repo_dir:
        return f"Error: Could not find repository '{repo_name}' anywhere in the codebase."

    results = []
    try:
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d != ".git"]
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            if query in line:
                                rel_path = os.path.relpath(file_path, repo_dir)
                                results.append(f"{rel_path}:{line_num}: {line.strip()}")
                except (UnicodeDecodeError, OSError):
                    pass

        if not results:
            return f"No exact matches found for '{query}'."

        if len(results) > 100:
            return (
                "\n".join(results[:100])
                + f"\n\n... and {len(results) - 100} more matches (showing first 100)."
            )

        return "\n".join(results)
    except Exception as e:
        return f"Error searching code: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 5: Semantic RAG Search (uses vector store)
# ---------------------------------------------------------------------------
@tool
def search_repo_rag(query: str) -> str:
    """
    Performs a semantic similarity search over the indexed repository using the
    vector store. Returns the most relevant code/documentation chunks.
    Use this tool when you need conceptual or fuzzy search (e.g. 'where is authentication handled?').

    Args:
        query (str): A natural language question or topic to search for semantically.
    """
    if not rag_manager.is_indexed():
        return (
            "No repository is currently indexed in the vector store. "
            "Please clone a repository first."
        )
    return rag_manager.query(query, k=5)


# ---------------------------------------------------------------------------
# Tool 6: Index a specific file or folder for RAG
# ---------------------------------------------------------------------------
@tool
def index_path_for_rag(path: str) -> str:
    """
    Indexes a specific file or folder from the local repos/ directory into the
    RAG (vector store) system. Use this when the user asks to 'use RAG for X',
    'index the X folder', 'load X into RAG', or 'focus on X file/folder'.

    After indexing, all subsequent semantic searches (search_repo_rag) will
    operate ONLY over this file/folder's content.

    Args:
        path (str): Either:
            - A path relative to the repos/ directory (e.g. 'my-repo/src' or 'my-repo/utils.py')
            - An absolute path to a file or folder on disk.

    Returns:
        A structured status string:
            INDEX_NAME:<name>|INDEX_PATH:<path>|<status message>
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repos_dir = os.path.join(base_dir, "repos")

    # Resolve the target path
    # 1. Try as an absolute path first
    if os.path.isabs(path) and os.path.exists(path):
        target_path = path
    else:
        # 2. Try relative to repos/
        candidate = os.path.join(repos_dir, path)
        if os.path.exists(candidate):
            target_path = candidate
        else:
            # 3. Search for the name anywhere inside repos/ using the shared helper
            found = find_in_repos(path, find_file=True, base_dir=base_dir)
            if found:
                target_path = found
            else:
                return (
                    f"[!] Could not locate '{path}' inside the repos/ directory. "
                    f"Use list_repo_files to browse available paths."
                )

    index_name = os.path.basename(target_path.rstrip("/\\"))
    status, _ = rag_manager.index_path(target_path)

    if status.startswith("[!]"):
        return status

    return (
        f"INDEX_NAME:{index_name}|"
        f"INDEX_PATH:{target_path}|"
        f"{status}"
    )

