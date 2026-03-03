"""Validate-and-repair loop for learning plans.

Orchestrates deterministic validation (checks.py) and LLM repair
(repair.py) in a loop until the plan passes all checks or max
iterations are exhausted. Updates DB status on completion.

validate_and_repair() is the Phase 8 entry point.
"""

import logging

logger = logging.getLogger(__name__)

MAX_REPAIR_ITERATIONS = 3


def validate_and_repair(
    path_run_id,
    plan_response,
    gap_report,
    retrieval_result,
    user_context,
    db_path=None,
):
    """Validate a learning plan and repair it if violations are found.

    Runs deterministic validation checks. If the plan fails, calls the
    LLM repair function and re-validates, up to MAX_REPAIR_ITERATIONS
    times. Updates learning_path_runs.status to 'validated' on success
    or 'failed' after max iterations exhausted.

    Args:
        path_run_id: UUID string from Phase 7 plan generation.
        plan_response: LearningPlanResponse from Phase 7.
        gap_report: SkillGapReport from Phase 5.
        retrieval_result: RetrievalResult from Phase 6.
        user_context: UserContext from Phase 5.
        db_path: Path to SQLite database. If None, uses config default.

    Returns:
        Tuple of (LearningPlanResponse, ValidationResult). Always returns
        a plan (best-effort on failure) and the final validation result.
    """
    from generation.plan_persistence import update_plan_status
    from generation.prereq_loader import (
        load_prerequisites_for_candidates,
    )
    from models.schemas import LearningPlanResponse, ValidationResult
    from validation.checks import validate_plan
    from validation.repair import repair_plan

    # Empty plan early exit
    if len(plan_response.courses) == 0:
        logger.info("Empty plan, skipping validation")
        update_plan_status(path_run_id, "validated", db_path)
        return (plan_response, ValidationResult(is_valid=True))

    # Load prereqs once for all iterations
    candidate_ids = [c.course_id for c in retrieval_result.candidates]
    prereqs_by_course = load_prerequisites_for_candidates(candidate_ids, db_path)

    # Extract budget from user context
    budget_hours = user_context.training_goal

    current_plan = plan_response
    result = None

    for iteration in range(MAX_REPAIR_ITERATIONS):
        result = validate_plan(current_plan, prereqs_by_course, gap_report, budget_hours)
        result.iteration = iteration

        if result.is_valid:
            update_plan_status(path_run_id, "validated", db_path)
            logger.info("Plan validated on iteration %d", iteration + 1)
            return (current_plan, result)

        # Count errors and warnings for logging
        error_count = sum(1 for v in result.violations if v.severity == "error")
        warning_count = sum(1 for v in result.violations if v.severity == "warning")
        logger.info(
            "Iteration %d: %d violations (%d errors, %d warnings), attempting repair",
            iteration + 1,
            len(result.violations),
            error_count,
            warning_count,
        )

        # Don't repair on the last iteration -- just report final result
        if iteration == MAX_REPAIR_ITERATIONS - 1:
            break

        # LLM repair
        repaired = repair_plan(
            current_plan,
            result,
            gap_report,
            retrieval_result,
            prereqs_by_course,
            budget_hours,
        )

        # Post-repair course ID re-validation: filter out hallucinated IDs
        valid_ids = {c.course_id for c in retrieval_result.candidates}
        original_count = len(repaired.courses)
        valid_courses = [c for c in repaired.courses if c.course_id in valid_ids]

        if len(valid_courses) < original_count:
            logger.warning(
                "Filtered %d hallucinated course IDs from repaired plan",
                original_count - len(valid_courses),
            )
            repaired = LearningPlanResponse(
                courses=valid_courses,
                total_estimated_hours=sum(
                    c.estimated_duration_hours for c in valid_courses
                ),
                skill_coverage_summary=repaired.skill_coverage_summary,
            )

        current_plan = repaired

    # Max iterations exhausted
    update_plan_status(path_run_id, "failed", db_path)
    logger.info(
        "Plan failed validation after %d iterations", MAX_REPAIR_ITERATIONS
    )
    return (current_plan, result)
