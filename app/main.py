import os
import uuid
from dotenv import load_dotenv

# Load .env before importing app modules (config.py reads env vars at module level)
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graph import create_graph
from app.rag import rag_manager
from app.chat_store import init_db, save_message


# ---------------------------------------------------------------------------
# Nodes whose LLM output is the final user-facing response.
# Tokens from these nodes are streamed to the terminal.
# Internal pipeline nodes (planner, grader, router, etc.) are silently skipped.
# "clone_repo" is included so the cancellation AIMessage is shown to the user.
# ---------------------------------------------------------------------------
_USER_FACING_NODES = {
    "chat",               # general conversation
    "chat_update",        # post-clone summary
    "generate",           # Self-CRAG answer generation
    "fallback",           # max-retries fallback message
    "direct_analyze",     # tool-based analysis answer
    "index_path_update",  # index confirmation
    "clone_repo",         # shows cancellation message when clone is declined
}


# ---------------------------------------------------------------------------
# Stream helper
# ---------------------------------------------------------------------------
def _stream(graph, input_or_command, config: dict) -> tuple[dict | None, str]:
    """
    Run graph.stream() and collect the last 'values' state + all streamed text.
    Delays printing 'Agent: ' until the first real token arrives.
    Returns (final_state, agent_response_text).
    """
    final_state    = None
    agent_response = ""
    header_printed = False

    for mode, data in graph.stream(
        input_or_command,
        config=config,
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            chunk, metadata = data
            node = metadata.get("langgraph_node", "")
            if node in _USER_FACING_NODES:
                content     = getattr(chunk, "content", "")
                tool_chunks = getattr(chunk, "tool_call_chunks", None)
                if content and not tool_chunks:
                    if not header_printed:
                        print("\nAgent: ", end="", flush=True)
                        header_printed = True
                    print(content, end="", flush=True)
                    agent_response += content

        elif mode == "values":
            final_state = data

    if header_printed:
        print()   # newline after streamed response

    return final_state, agent_response


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    # Initialise local SQLite store (creates chat_history.db if needed)
    init_db()

    # MemorySaver enables interrupt/resume without any external database.
    checkpointer = MemorySaver()
    thread_id    = str(uuid.uuid4())
    config       = {"configurable": {"thread_id": thread_id}}

    print(f"\n  GitHub Repo Agent  |  session: {thread_id[:8]}…  |  type 'exit' to quit\n")

    graph = create_graph(checkpointer=checkpointer)

    repo_name:     str  = ""
    repo_metadata: dict = {}

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                rag_manager.clear()
                print("Goodbye!")
                break

            # Save user message to SQLite
            save_message(thread_id, "user", user_input)

            # Pass ONLY the new HumanMessage — MemorySaver accumulates history
            initial_state = {
                "messages":         [HumanMessage(content=user_input)],
                "intent":           "",
                "repo_name":        repo_name,
                "repo_metadata":    repo_metadata,
                # Planner fields — reset each turn
                "plan":             "",
                "planner_query":    "",
                "tool_hints":       [],
                # Self-CRAG fields — reset each turn
                "question":         user_input,
                "documents":        [],
                "generation":       "",
                "loop_count":       0,
                "retrieval_needed": True,
            }

            final_state, agent_response = _stream(graph, initial_state, config)

            # ── Check for human-in-the-loop interrupt ────────────────────────
            graph_state = graph.get_state(config)
            interrupted = (
                graph_state.tasks
                and any(getattr(t, "interrupts", None) for t in graph_state.tasks)
            )

            if interrupted:
                # Retrieve the interrupt message set by the clone node
                interrupt_msg = ""
                for task in graph_state.tasks:
                    interrupts = getattr(task, "interrupts", None)
                    if interrupts:
                        interrupt_msg = interrupts[0].value
                        break

                # Show the confirmation prompt
                print(f"\n{interrupt_msg}")
                human_answer = input("\n  Your answer: ").strip()

                # Resume the graph with the user's answer
                final_state, resume_response = _stream(
                    graph, Command(resume=human_answer), config
                )
                agent_response = resume_response

            # Save assistant response to SQLite
            if agent_response:
                save_message(thread_id, "assistant", agent_response)

            # ── Sync local state from graph result ───────────────────────────
            if final_state:
                if final_state.get("repo_name"):
                    repo_name = final_state["repo_name"]
                if final_state.get("repo_metadata"):
                    repo_metadata = final_state["repo_metadata"]
                elif rag_manager.current_metadata:
                    repo_metadata = rag_manager.current_metadata

        except KeyboardInterrupt:
            print()
            rag_manager.clear()
            print("Goodbye!")
            break
        except Exception as e:
            print(f"\n[!] Error: {e}")


if __name__ == "__main__":
    main()
