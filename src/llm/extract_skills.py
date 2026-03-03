"""Batch skill extraction from course summaries using LLM structured output.

Provides two public functions:
- extract_skills_for_course(): Extract skills for a single course (with summary)
- process_all_courses(): Batch process all unprocessed courses

Supports two extraction paths:
- Full extraction: uses course name + summary_text (higher confidence)
- Name-only fallback: uses course name alone for courses without a summary
"""

import logging
from typing import cast

logger = logging.getLogger(__name__)


def extract_skills_for_course(
    course_name: str, summary_text: str, model=None
):
    """Extract skills from a single course summary using LLM.

    Args:
        course_name: Name of the course.
        summary_text: Course summary/description text.
        model: Optional LangChain chat model. Created via get_chat_model() if None.

    Returns:
        CourseSkillsResponse with extracted skills, or None if the LLM
        refused or an API error occurred.
    """
    from langchain_core.prompts import ChatPromptTemplate

    from llm.client import get_chat_model
    from llm.prompts import SKILL_EXTRACTION_PROMPT
    from models.schemas import CourseSkillsResponse

    if model is None:
        model = get_chat_model()

    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SKILL_EXTRACTION_PROMPT),
                ("user", "Course: {course_name}\n\nSummary: {summary_text}"),
            ]
        )

        structured_model = model.with_structured_output(CourseSkillsResponse)
        chain = prompt | structured_model
        result = cast(
            CourseSkillsResponse,
            chain.invoke({"course_name": course_name, "summary_text": summary_text}),
        )

        if result is None:
            logger.warning(
                "LLM refusal for course '%s': no parsed output", course_name
            )
            return None

        return result

    except Exception:
        logger.warning(
            "API error extracting skills for course '%s'",
            course_name,
            exc_info=True,
        )
        return None


def extract_skills_from_name_only(course_name: str, model=None):
    """Extract skills from a course name alone (no summary available).

    Uses a conservative prompt that caps skill_confidence at 0.7 to signal
    lower certainty on name-only inferences.

    Args:
        course_name: Name of the course.
        model: Optional LangChain chat model. Created via get_chat_model() if None.

    Returns:
        CourseSkillsResponse with extracted skills, or None on error.
    """
    from langchain_core.prompts import ChatPromptTemplate

    from llm.client import get_chat_model
    from llm.prompts import NAME_ONLY_SKILL_EXTRACTION_PROMPT
    from models.schemas import CourseSkillsResponse

    if model is None:
        model = get_chat_model()

    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", NAME_ONLY_SKILL_EXTRACTION_PROMPT),
                ("user", "Course: {course_name}"),
            ]
        )

        structured_model = model.with_structured_output(CourseSkillsResponse)
        chain = prompt | structured_model
        result = cast(
            CourseSkillsResponse,
            chain.invoke({"course_name": course_name}),
        )

        if result is None:
            logger.warning(
                "LLM refusal for name-only course '%s': no parsed output", course_name
            )
            return None

        return result

    except Exception:
        logger.warning(
            "API error extracting skills (name-only) for course '%s'",
            course_name,
            exc_info=True,
        )
        return None


def process_all_courses(db_path=None, model=None, force: bool = False):
    """Batch extract skills for all unprocessed courses.

    Iterates over all courses that haven't been processed yet. For courses
    with a summary, uses full extraction. For courses without a summary,
    falls back to name-only extraction with a conservative prompt.
    Each course is committed atomically (skills + processed_by_llm flag).

    Args:
        db_path: Optional database path. Uses config default if None.
        model: Optional LangChain chat model. Created once if None.
        force: If True, reset all processing state and re-extract everything.

    Returns:
        Dict with keys: processed, skipped, total_skills, total_courses.
    """
    from db.connection import get_connection
    from llm.client import get_chat_model

    conn = get_connection(db_path)

    if model is None:
        model = get_chat_model()

    try:
        if force:
            conn.execute("UPDATE courses SET processed_by_llm = 0")
            conn.execute("DELETE FROM course_skills")
            conn.commit()
            logger.info("Force mode: reset all processing state")

        rows = conn.execute(
            "SELECT course_id, course_name, summary_text FROM courses "
            "WHERE processed_by_llm = 0"
        ).fetchall()

        total_courses = len(rows)
        logger.info("Found %d courses to process", total_courses)

        processed = 0
        skipped = 0
        total_skills = 0

        for row in rows:
            course_id = row["course_id"]
            course_name = row["course_name"]
            summary_text = row["summary_text"]

            has_summary = summary_text and summary_text.strip()

            if has_summary:
                result = extract_skills_for_course(
                    course_name, summary_text, model
                )
            else:
                logger.debug(
                    "Course %d ('%s') has no summary — using name-only extraction",
                    course_id,
                    course_name,
                )
                result = extract_skills_from_name_only(course_name, model)

            if result is None:
                logger.warning(
                    "Skipping course %d ('%s'): extraction returned None",
                    course_id,
                    course_name,
                )
                skipped += 1
                continue

            conn.execute(
                "DELETE FROM course_skills WHERE course_id = ?", (course_id,)
            )

            skills_data = [
                (
                    course_id,
                    skill.skill_name,
                    skill.skill_level,
                    skill.skill_confidence,
                    skill.topic_name,
                    skill.topic_weight,
                )
                for skill in result.skills
            ]

            if skills_data:
                conn.executemany(
                    "INSERT INTO course_skills "
                    "(course_id, skill_name, skill_level, skill_confidence, "
                    "topic_name, topic_weight) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    skills_data,
                )

            conn.execute(
                "UPDATE courses SET processed_by_llm = 1, "
                "processed_at = datetime('now') WHERE course_id = ?",
                (course_id,),
            )
            conn.commit()

            num_skills = len(result.skills)
            total_skills += num_skills
            processed += 1
            extraction_type = "summary" if has_summary else "name-only"
            logger.info(
                "[%d/%d] Processed course %d ('%s'): %d skills [%s]",
                processed + skipped,
                total_courses,
                course_id,
                course_name,
                num_skills,
                extraction_type,
            )

        return {
            "processed": processed,
            "skipped": skipped,
            "total_skills": total_skills,
            "total_courses": total_courses,
        }

    finally:
        conn.close()
