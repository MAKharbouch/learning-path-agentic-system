"""LLM-driven learning plan generation.

Takes skill gaps + candidate courses + user context and produces a
sequenced learning plan via LLM structured output. Persists the
result to SQLite as a draft plan for Phase 8 validation.

generate_learning_plan() is the Phase 7 entry point.
"""

import logging
import uuid
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_HOURS = 20.0


def _build_prompt_context(gap_report, retrieval_result, prereqs_by_course):
    """Format skill gaps, candidates, and prereqs into prompt text blocks.

    Args:
        gap_report: A SkillGapReport.
        retrieval_result: A RetrievalResult with candidates.
        prereqs_by_course: Dict mapping course_id to list of prereq dicts.

    Returns:
        Tuple of (skill_gaps_text, candidates_text).
    """
    # Format skill gaps
    gap_lines = []
    for i, gap in enumerate(gap_report.gaps, 1):
        current = gap.current_level or "none"
        gap_lines.append(
            f"{i}. {gap.skill_name} — required: {gap.required_level}, "
            f"current: {current}, type: {gap.gap_type}, priority: {gap.priority}"
        )
    skill_gaps_text = "\n".join(gap_lines) if gap_lines else "No skill gaps identified."

    # Format candidates with prereqs and summary
    candidate_lines = []
    for c in retrieval_result.candidates:
        # Truncate summary to 300 chars to manage token budget
        summary = c.summary_text or "No summary available"
        if len(summary) > 300:
            summary = summary[:297] + "..."

        skills_str = ", ".join(c.matched_skills) if c.matched_skills else "none"

        line = (
            f"- Course ID: {c.course_id} | Name: {c.course_name} | "
            f"Score: {c.score:.4f} | Source: {c.source}\n"
            f"  Matched skills: {skills_str}\n"
            f"  Summary: {summary}"
        )

        # Add prerequisites if any
        course_prereqs = prereqs_by_course.get(c.course_id, [])
        if course_prereqs:
            prereq_names = [p["prereq_name"] for p in course_prereqs]
            line += f"\n  Prerequisites: {', '.join(prereq_names)}"
        else:
            line += "\n  Prerequisites: none"

        candidate_lines.append(line)

    candidates_text = "\n\n".join(candidate_lines) if candidate_lines else "No candidates available."

    return skill_gaps_text, candidates_text


def generate_learning_plan(
    gap_report,
    retrieval_result,
    user_context,
    db_path: Path | None = None,
):
    """Generate a learning plan from skill gaps and candidate courses.

    This is the Phase 7 entry point. It:
    1. Loads prerequisite data for all candidate courses
    2. Builds the prompt context (skill gaps, candidates with prereqs)
    3. Calls the LLM with structured output to generate the plan
    4. Validates that returned course IDs exist in the candidate list
    5. Persists the plan to SQLite as a draft

    Args:
        gap_report: A SkillGapReport from Phase 5.
        retrieval_result: A RetrievalResult from Phase 6.
        user_context: A UserContext from Phase 5.
        db_path: Path to SQLite database. If None, uses config default.

    Returns:
        Tuple of (path_run_id, LearningPlanResponse).
        path_run_id is a UUID4 string identifying this plan run.
    """
    from langchain_core.prompts import ChatPromptTemplate

    from generation.plan_persistence import save_plan_to_db
    from generation.prereq_loader import (
        load_prerequisites_for_candidates,
    )
    from llm.prompts import PLAN_GENERATION_PROMPT
    from models.schemas import LearningPlanResponse

    # Handle empty candidates early
    if not retrieval_result.candidates:
        logger.info("No candidates available, returning empty plan")
        empty_plan = LearningPlanResponse(
            courses=[],
            total_estimated_hours=0.0,
            skill_coverage_summary="No candidate courses available to address skill gaps.",
        )
        path_run_id = str(uuid.uuid4())
        save_plan_to_db(
            path_run_id=path_run_id,
            portal_id=user_context.portal_id,
            training_goal_hours=user_context.training_goal,
            plan_response=empty_plan,
            db_path=db_path,
        )
        return (path_run_id, empty_plan)

    # Step 1: Load prerequisite data
    candidate_ids = [c.course_id for c in retrieval_result.candidates]
    prereqs_by_course = load_prerequisites_for_candidates(candidate_ids, db_path)

    # Step 2: Build prompt context
    skill_gaps_text, candidates_text = _build_prompt_context(
        gap_report, retrieval_result, prereqs_by_course
    )

    # Determine budget
    budget_hours = user_context.training_goal
    if budget_hours is None:
        budget_hours_str = "unconstrained"
        logger.info("No training goal set, using unconstrained budget")
    else:
        budget_hours_str = str(budget_hours)
        logger.info("Training hour budget: %s hours", budget_hours_str)

    # Step 3: LLM call with structured output
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLAN_GENERATION_PROMPT),
            (
                "user",
                "Generate a learning plan for user {portal_id} based on the "
                "skill gaps and candidate courses provided in the system prompt.",
            ),
        ]
    )

    from llm.client import get_chat_model

    model = get_chat_model()
    structured_model = model.with_structured_output(LearningPlanResponse)

    chain = prompt | structured_model
    plan_response = cast(
        LearningPlanResponse,
        chain.invoke(
            {
                "budget_hours": budget_hours_str,
                "skill_gaps": skill_gaps_text,
                "candidates": candidates_text,
                "portal_id": user_context.portal_id,
            }
        ),
    )

    logger.info(
        "LLM generated plan with %d courses, %.1f total hours",
        len(plan_response.courses),
        plan_response.total_estimated_hours,
    )

    # Step 4: Validate course IDs (filter out hallucinated IDs)
    valid_ids = {c.course_id for c in retrieval_result.candidates}
    original_count = len(plan_response.courses)
    valid_courses = [c for c in plan_response.courses if c.course_id in valid_ids]

    if len(valid_courses) < original_count:
        logger.warning(
            "Filtered %d hallucinated course IDs from plan",
            original_count - len(valid_courses),
        )
        # Rebuild response with valid courses only
        plan_response = LearningPlanResponse(
            courses=valid_courses,
            total_estimated_hours=sum(
                c.estimated_duration_hours for c in valid_courses
            ),
            skill_coverage_summary=plan_response.skill_coverage_summary,
        )

    # Step 5: Persist to DB as draft
    path_run_id = str(uuid.uuid4())
    save_plan_to_db(
        path_run_id=path_run_id,
        portal_id=user_context.portal_id,
        training_goal_hours=user_context.training_goal,
        plan_response=plan_response,
        db_path=db_path,
    )

    logger.info("Plan persisted as draft: %s", path_run_id)

    return (path_run_id, plan_response)
