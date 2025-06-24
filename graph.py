from langgraph.graph import StateGraph
from typing import TypedDict
from langgraph_nodes import inference_node, confidence_check_node, fallback_node


# Define your state schema
class EmotionState(TypedDict, total=False):
    input: str
    original_input: str
    probs: list
    initial_labels: list
    max_conf: float
    fallback_needed: bool
    clarified: bool
    final_labels: list
    fallback_method: str
    clarified_input: str
    # Bonus features
    backup_labels: list
    backup_conf: float
    backup_model_used: bool
    primary_model_used: bool


def build_graph():
    builder = StateGraph(state_schema=EmotionState)

    builder.add_node("inference", inference_node)
    builder.add_node("check_confidence", confidence_check_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("final", lambda state: state)

    builder.set_entry_point("inference")
    builder.add_edge("inference", "check_confidence")

    # Fixed: Remove the problematic edge that causes the loop
    # The fallback node should go directly to final, not back to check_confidence
    builder.add_conditional_edges(
        "check_confidence",
        lambda state: "fallback" if state.get("fallback_needed") and not state.get("clarified") else "final"
    )

    # Fixed: fallback goes to final, not back to check_confidence
    builder.add_edge("fallback", "final")

    return builder.compile()