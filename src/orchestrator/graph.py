"""LangGraph state graph for the learning path pipeline.

build_graph() constructs and compiles the graph.
run_pipeline() is the public entry point for end-to-end plan generation.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from orchestrator.errors import MAX_REPAIRS
from orchestrator.nodes import (
    analyze_node,
    generate_node,
    repair_node,
    retrieve_node,
    validate_node,
)
from orchestrator.state import InputState, OutputState, PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing functions (pure -- no side effects, no DB calls)
# ---------------------------------------------------------------------------


def route_after_analyze(state: PipelineState) -> Literal["retrieve", "__end__"]:
    """Route after analyze: continue to retrieve, or end on error / no gaps."""
    if state.get("error"):
        return "__end__"
    gap_report = state.get("gap_report")
    if gap_report is not None and len(gap_report.gaps) == 0:
        return "__end__"
    return "retrieve"


def route_after_retrieve(state: PipelineState) -> Literal["generate", "__end__"]:
    """Route after retrieve: continue to generate, or end on error."""
    if state.get("error"):
        return "__end__"
    return "generate"


def route_after_generate(state: PipelineState) -> Literal["validate", "__end__"]:
    """Route after generate: continue to validate, or end on error."""
    if state.get("error"):
        return "__end__"
    return "validate"


def route_after_validate(state: PipelineState) -> Literal["repair", "finalize"]:
    """Route after validate: repair if invalid and under limit, else finalize."""
    if state.get("error"):
        return "finalize"
    validation_result = state.get("validation_result")
    if validation_result is not None and validation_result.is_valid:
        return "finalize"
    if state.get("repair_count", 0) >= MAX_REPAIRS:
        return "finalize"
    return "repair"


def route_after_repair(state: PipelineState) -> Literal["validate", "__end__"]:
    """Route after repair: re-validate, or end on error."""
    if state.get("error"):
        return "__end__"
    return "validate"


# ---------------------------------------------------------------------------
# Finalize node (DB status updates -- the only node with side effects on DB)
# ---------------------------------------------------------------------------


def finalize_node(state: PipelineState) -> dict:
    """Update DB plan status based on validation result.

    This node is the single place where plan status transitions from
    'draft' to 'validated' or 'failed'.  It is deliberately lenient:
    errors are logged but never propagated, because the plan data is
    already captured in state and losing the status update is
    preferable to crashing the pipeline.
    """
    try:
        if state.get("error"):
            # Plan may not exist in DB yet (e.g. analyze failed before
            # generate could persist), so we skip the status update.
            return {}

        from generation.plan_persistence import update_plan_status

        validation_result = state.get("validation_result")
        path_run_id = state.get("path_run_id")

        if not path_run_id:
            logger.warning("finalize_node: no path_run_id in state, skipping status update")
            return {}

        if validation_result is not None and validation_result.is_valid:
            update_plan_status(path_run_id, "validated")
        else:
            # Max repairs exhausted or validation still invalid
            update_plan_status(path_run_id, "failed")

    except Exception as exc:
        logger.warning("finalize_node: failed to update plan status — %s", exc)

    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph():
    """Construct and compile the LangGraph state graph.

    Returns a compiled ``CompiledStateGraph`` with six nodes
    (analyze, retrieve, generate, validate, repair, finalize),
    conditional routing at each step, and retry policies on
    LLM-calling nodes.
    """
    builder = StateGraph(PipelineState, input_schema=InputState, output_schema=OutputState)

    llm_retry = RetryPolicy(max_attempts=3)

    # Nodes -----------------------------------------------------------------
    builder.add_node("analyze", analyze_node, retry_policy=llm_retry)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node, retry_policy=llm_retry)
    builder.add_node("validate", validate_node)
    builder.add_node("repair", repair_node, retry_policy=llm_retry)
    builder.add_node("finalize", finalize_node)

    # Edges -----------------------------------------------------------------
    builder.add_edge(START, "analyze")
    builder.add_conditional_edges("analyze", route_after_analyze)
    builder.add_conditional_edges("retrieve", route_after_retrieve)
    builder.add_conditional_edges("generate", route_after_generate)
    builder.add_conditional_edges("validate", route_after_validate)
    builder.add_conditional_edges("repair", route_after_repair)
    builder.add_edge("finalize", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_pipeline(portal_id: int, goal_text: str) -> dict:
    """Run the full learning path pipeline for a user.

    This is the public entry point that Phase 10 (Streamlit UI) will
    call.  It builds the graph, invokes it with the given inputs,
    and returns the result dict (filtered to OutputState fields by
    LangGraph).

    Args:
        portal_id: The user's portal ID.
        goal_text: Free-text learning goal description.

    Returns:
        A dict with OutputState fields: plan_response,
        validation_result, path_run_id, and optionally error /
        error_node.
    """
    logger.info("run_pipeline: starting for portal_id=%s", portal_id)

    graph = build_graph()
    result = graph.invoke(
        {"portal_id": portal_id, "goal_text": goal_text},
        {"recursion_limit": 25},
    )

    logger.info("run_pipeline: complete for portal_id=%s", portal_id)
    return result
