from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # 'messages' holds the full conversation history.
    # add_messages appends new messages rather than overwriting.
    messages: Annotated[list[BaseMessage], add_messages]

    # 'intent' holds the classifier's decision for the current turn.
    # No reducer → LangGraph overwrites with the latest value each turn.
    intent: str

    # 'repo_name' holds the name of the currently active cloned repository.
    # Populated by index_repo_node after a successful clone + index.
    # Persists across turns so the user doesn't have to repeat the repo name.
    repo_name: str

    # -----------------------------------------------------------------------
    # Repo Metadata — loaded from .meta/repo_metadata.json after cloning.
    # Persists across turns; provides the Planner and Generator with high-level
    # repo context (overview, main language, key files).
    # -----------------------------------------------------------------------
    repo_metadata: dict

    # -----------------------------------------------------------------------
    # Self-CRAG fields — populated and consumed within the analyze_repo branch
    # -----------------------------------------------------------------------

    # The user's question extracted from messages, passed through the CRAG pipeline.
    question: str

    # -----------------------------------------------------------------------
    # Planner fields — populated by planner_node before the CRAG pipeline.
    # -----------------------------------------------------------------------

    # Step-by-step reasoning / plan produced by the Planner LLM.
    # Injected as context into generate_node.
    plan: str

    # Reformulated, vector-search-optimised query produced by the Planner.
    # Replaces 'question' for retrieve_node and rewrite_query_node.
    planner_query: str

    # Tool hints suggested by the Planner (list of tool name strings).
    tool_hints: list

    # Raw Document chunks returned by the retriever (before grading).
    documents: list[Document]

    # Draft answer produced by the generate_node before reflection.
    generation: str

    # How many retrieval+rewrite cycles have been attempted this turn.
    # Prevents infinite loops — graph exits after MAX_RETRIES attempts.
    loop_count: int

    # Decision made by the retrieval_router_node:
    #   True  → query requires semantic retrieval from the vector DB
    #   False → query can be answered with parametric knowledge alone
    retrieval_needed: bool
