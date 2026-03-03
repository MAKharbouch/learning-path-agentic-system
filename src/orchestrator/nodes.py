"""Node wrapper functions for the LangGraph pipeline.

Each node is a thin wrapper around existing Phase 5-8 entry points,
adding error capture and state packing/unpacking.
"""

import logging

from orchestrator.state import PipelineState

logger = logging.getLogger(__name__)


def analyze_node(state: PipelineState) -> dict:
    """Phase 5 wrapper: user context + goal skill extraction + gap analysis.

    Calls ``analyze_skill_gap(portal_id, goal_text)`` and unpacks the
    3-tuple into individual state fields.
    """
    logger.info("analyze_node: starting for portal_id=%s", state["portal_id"])
    try:
        from analysis.skill_gap import analyze_skill_gap

        user_context, goal_skills, gap_report = analyze_skill_gap(
            state["portal_id"], state["goal_text"]
        )

        logger.info(
            "analyze_node: complete — %d gaps identified",
            len(gap_report.gaps),
        )
        return {
            "user_context": user_context,
            "goal_skills": goal_skills,
            "gap_report": gap_report,
        }
    except Exception as e:
        logger.exception("analyze_node: failed — %s", e)
        return {"error": str(e), "error_node": "analyze"}


def retrieve_node(state: PipelineState) -> dict:
    """Phase 6 wrapper: hybrid course retrieval (SQL + semantic via RRF).

    Calls ``retrieve_candidates(gap_report, user_context)`` and stores
    the RetrievalResult in state.
    """
    logger.info("retrieve_node: starting")
    try:
        from retrieval.hybrid import retrieve_candidates

        result = retrieve_candidates(state["gap_report"], state["user_context"])

        logger.info(
            "retrieve_node: complete — %d candidates",
            len(result.candidates),
        )
        return {"retrieval_result": result}
    except Exception as e:
        logger.exception("retrieve_node: failed — %s", e)
        return {"error": str(e), "error_node": "retrieve"}


def generate_node(state: PipelineState) -> dict:
    """Phase 7 wrapper: LLM plan generation + prereq loading.

    Calls ``generate_learning_plan()`` for the plan, then loads
    prerequisites once via ``load_prerequisites_for_candidates()``
    and stores them in state for downstream validate/repair reuse.
    """
    logger.info("generate_node: starting")
    try:
        from generation.plan_generator import generate_learning_plan
        from generation.prereq_loader import (
            load_prerequisites_for_candidates,
        )

        path_run_id, plan_response = generate_learning_plan(
            state["gap_report"],
            state["retrieval_result"],
            state["user_context"],
        )

        # Load prereqs once; validate_node and repair_node will read from state
        candidate_ids = [
            c.course_id for c in state["retrieval_result"].candidates
        ]
        prereqs_by_course = load_prerequisites_for_candidates(candidate_ids)

        logger.info(
            "generate_node: complete — plan %s with %d courses, prereqs for %d candidates",
            path_run_id,
            len(plan_response.courses),
            len(prereqs_by_course),
        )
        return {
            "path_run_id": path_run_id,
            "plan_response": plan_response,
            "prereqs_by_course": prereqs_by_course,
            "repair_count": 0,
        }
    except Exception as e:
        logger.exception("generate_node: failed — %s", e)
        return {"error": str(e), "error_node": "generate"}


def validate_node(state: PipelineState) -> dict:
    """Phase 8 validation wrapper: deterministic plan checks.

    Reads ``prereqs_by_course`` from state (loaded once in generate_node)
    instead of redundantly querying the database.  For empty plans,
    short-circuits to a valid result and updates DB status.
    """
    logger.info("validate_node: starting (repair_count=%s)", state.get("repair_count", 0))
    try:
        from validation.checks import validate_plan

        plan_response = state["plan_response"]

        # Empty plan fast path
        if not plan_response.courses:
            from generation.plan_persistence import update_plan_status
            from models.schemas import ValidationResult

            update_plan_status(state["path_run_id"], "validated")
            logger.info("validate_node: empty plan — auto-validated")
            return {"validation_result": ValidationResult(is_valid=True)}

        result = validate_plan(
            plan_response,
            state["prereqs_by_course"],
            state["gap_report"],
            state["user_context"].training_goal,
        )
        result.iteration = state.get("repair_count", 0)

        logger.info(
            "validate_node: complete — is_valid=%s, %d violations",
            result.is_valid,
            len(result.violations),
        )
        return {"validation_result": result}
    except Exception as e:
        logger.exception("validate_node: failed — %s", e)
        return {"error": str(e), "error_node": "validate"}


def repair_node(state: PipelineState) -> dict:
    """Phase 8 repair wrapper: LLM-driven plan correction.

    Calls ``repair_plan()`` with the current plan, violations, and
    original context.  Applies a post-repair hallucination guard
    (filters course IDs not in the candidate list) and increments
    ``repair_count``.
    """
    count = state.get("repair_count", 0)
    logger.info("repair_node: starting (repair_count=%d)", count)
    try:
        from models.schemas import LearningPlanResponse
        from validation.repair import repair_plan

        repaired = repair_plan(
            state["plan_response"],
            state["validation_result"],
            state["gap_report"],
            state["retrieval_result"],
            state["prereqs_by_course"],
            state["user_context"].training_goal,
        )

        # Post-repair hallucination guard: filter to valid candidate IDs
        valid_ids = {c.course_id for c in state["retrieval_result"].candidates}
        original_count = len(repaired.courses)
        valid_courses = [c for c in repaired.courses if c.course_id in valid_ids]

        if len(valid_courses) < original_count:
            logger.warning(
                "repair_node: filtered %d hallucinated course IDs",
                original_count - len(valid_courses),
            )
            repaired = LearningPlanResponse(
                courses=valid_courses,
                total_estimated_hours=sum(
                    c.estimated_duration_hours for c in valid_courses
                ),
                skill_coverage_summary=repaired.skill_coverage_summary,
            )

        logger.info(
            "repair_node: complete — %d courses after repair",
            len(repaired.courses),
        )
        return {"plan_response": repaired, "repair_count": count + 1}
    except Exception as e:
        logger.exception("repair_node: failed — %s", e)
        return {"error": str(e), "error_node": "repair"}
