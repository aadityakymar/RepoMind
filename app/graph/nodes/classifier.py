"""app/graph/nodes/classifier.py — Intent classification node."""

from app.state import AgentState
from app.graph.schemas import Intent


def make_classifier_node(llm):
    """Returns a classifier node that uses structured output to detect intent."""

    def classifier(state: AgentState):
        """Uses structured output to classify intent from the latest message."""
        structured_llm = llm.with_structured_output(Intent)
        result = structured_llm.invoke(state["messages"])
        return {"intent": result.intent}

    return classifier
