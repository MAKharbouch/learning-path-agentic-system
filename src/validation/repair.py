"""LLM-driven repair of learning plans that failed validation.

Takes a failed plan + structured violations + original context and
calls the LLM to produce a corrected LearningPlanResponse.
"""

import logging
from typing import cast

logger = logging.getLogger(__name__)


def repair_plan(
    plan_response,
    validation_result,
    gap_report,
    retrieval_result,
    prereqs_by_course,
    budget_hours,
):
    """Repair a failed learning plan using LLM structured output.

    Takes the current (invalid) plan, the validation violations, and the
    original generation context, then asks the LLM to produce a corrected
    LearningPlanResponse that addresses all violations.

    Args:
        plan_response: The failed LearningPlanResponse.
        validation_result: ValidationResult with violations from checks.
        gap_report: Original SkillGapReport from Phase 5.
        retrieval_result: Original RetrievalResult (candidates) from Phase 6.
        prereqs_by_course: Dict[int, list[dict]] prereq context.
        budget_hours: Float or None from user_context.training_goal.

    Returns:
        A repaired LearningPlanResponse from the LLM.
    """
    from langchain_core.prompts import ChatPromptTemplate

    from generation.plan_generator import _build_prompt_context
    from llm.prompts import PLAN_REPAIR_PROMPT
    from models.schemas import LearningPlanResponse

    # Format violations with error/warning separation
    errors = [v for v in validation_result.violations if v.severity == "error"]
    warnings = [v for v in validation_result.violations if v.severity == "warning"]

    violation_lines = []
    if errors:
        violation_lines.append("ERRORS (must fix):")
        for v in errors:
            violation_lines.append(f"- [{v.check_name}] {v.message}")
    if warnings:
        violation_lines.append("WARNINGS (should fix if possible):")
        for v in warnings:
            violation_lines.append(f"- [{v.check_name}] {v.message}")
    violations_text = "\n".join(violation_lines)

    # Reuse prompt context builder from plan_generator
    skill_gaps_text, candidates_text = _build_prompt_context(
        gap_report, retrieval_result, prereqs_by_course
    )

    # Format current plan for context
    plan_lines = []
    for course in sorted(plan_response.courses, key=lambda c: c.course_order):
        plan_lines.append(
            f"{course.course_order}. [{course.course_id}] {course.course_name} "
            f"({course.estimated_duration_hours}h, phase: {course.phase})"
        )
    current_plan_text = "\n".join(plan_lines) if plan_lines else "Empty plan."

    # Budget string
    budget_str = "unconstrained" if budget_hours is None else str(budget_hours)

    # Build prompt and model
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLAN_REPAIR_PROMPT),
            ("user", "Repair the learning plan based on the violations described."),
        ]
    )

    from llm.client import get_chat_model

    model = get_chat_model()
    structured_model = model.with_structured_output(LearningPlanResponse)

    chain = prompt | structured_model
    repaired = cast(
        LearningPlanResponse,
        chain.invoke(
            {
                "budget_hours": budget_str,
                "skill_gaps": skill_gaps_text,
                "candidates": candidates_text,
                "current_plan": current_plan_text,
                "violations": violations_text,
            }
        ),
    )

    logger.info(
        "LLM repaired plan: %d courses, %.1f total hours",
        len(repaired.courses),
        repaired.total_estimated_hours,
    )

    return repaired
