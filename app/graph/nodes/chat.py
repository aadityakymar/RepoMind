"""app/graph/nodes/chat.py — General conversation node."""

from app.state import AgentState
from app.prompts import CHAT_SYSTEM_PROMPT


def make_chat_node(llm):
    """Returns a chat node for general conversational responses."""

    def chat_node(state: AgentState):
        """Handles general conversational responses."""
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    return chat_node
