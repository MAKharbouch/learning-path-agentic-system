"""Batch prerequisite extraction from course summaries using LLM structured output.

Provides two public functions:
- extract_prerequisites_for_course(): Extract prerequisites for a single course
- process_all_prerequisites(): Batch process all unprocessed courses

Supports two extraction paths:
- Full extraction: uses course name + summary_text (higher confidence)
- Name-only fallback: uses course name alone for courses without a summary
"""

import logging
from typing import cast

logger = logging.getLogger(__name__)


def extract_prerequisites_for_course(
    course_name: str, summary_text: str, model=None
):
    """Extract prerequisites from a single course summary using LLM.

    Args:
        course_name: Name of the course.
        summary_text: Course summary/description text.
        model: Optional LangChain chat model. Created via get_chat_model() if None.

    Returns:
        CoursePrerequisitesResponse with extracted prerequisites, or None if
        the LLM refused or an API error occurred.
    """
    from langchain_core.prompts import ChatPromptTemplate

    from llm.client import get_chat_model
    from llm.prompts import PREREQUISITE_EXTRACTION_PROMPT
    from models.schemas import CoursePrerequisitesResponse

    if model is None:
        model = get_chat_model()

    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PREREQUISITE_EXTRACTION_PROMPT),
                ("user", "Course: {course_name}\n\nSummary: {summary_text}"),
            ]
        )

        structured_model = model.with_structured_output(CoursePrerequisitesResponse)
        chain = prompt | structured_model
        result = cast(
            CoursePrerequisitesResponse,
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
            "API error extracting prerequisites for course '%s'",
            course_name,
            exc_info=True,
        )
        return None


def extract_prerequisites_from_name_only(course_name: str, model=None):
    """Extract prerequisites from a course name alone (no summary available).

    Uses a conservative prompt that caps prereq_confidence at 0.6 to signal
    lower certainty on name-only inferences.

    Args:
        course_name: Name of the course.
        model: Optional LangChain chat model. Created via get_chat_model() if None.

    Returns:
        CoursePrerequisitesResponse with extracted prerequisites, or None on error.
    """
    from langchain_core.prompts import ChatPromptTemplate

    from llm.client import get_chat_model
    from llm.prompts import NAME_ONLY_PREREQUISITE_EXTRACTION_PROMPT
    from models.schemas import CoursePrerequisitesResponse

    if model is None:
        model = get_chat_model()

    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", NAME_ONLY_PREREQUISITE_EXTRACTION_PROMPT),
                ("user", "Course: {course_name}"),
            ]
        )

        structured_model = model.with_structured_output(CoursePrerequisitesResponse)
        chain = prompt | structured_model
        result = cast(
            CoursePrerequisitesResponse,
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
            "API error extracting prerequisites (name-only) for course '%s'",
            course_name,
            exc_info=True,
        )
        return None


def process_all_prerequisites(db_path=None, model=None, force: bool = False):
    """Batch extract prerequisites for all unprocessed courses.

    Iterates over all courses that haven't had prerequisites extracted yet.
    For courses with a summary, uses full extraction. For courses without a
    summary, falls back to name-only extraction with a conservative prompt.
    Each course is committed atomically (prerequisites + prereqs_extracted flag).

    Args:
        db_path: Optional database path. Uses config default if None.
        model: Optional LangChain chat model. Created once if None.
        force: If True, reset all processing state and re-extract everything.

    Returns:
        Dict with keys: processed, skipped, total_prerequisites, total_courses.
    """
    from db.connection import get_connection
    from llm.client import get_chat_model

    conn = get_connection(db_path)

    if model is None:
        model = get_chat_model()

    try:
        if force:
            conn.execute("UPDATE courses SET prereqs_extracted = 0")
            conn.execute("DELETE FROM course_prerequisites")
            conn.commit()
            logger.info("Force mode: reset all prerequisite processing state")

        rows = conn.execute(
            "SELECT course_id, course_name, summary_text FROM courses "
            "WHERE prereqs_extracted = 0"
        ).fetchall()

        total_courses = len(rows)
        logger.info("Found %d courses to process for prerequisites", total_courses)

        processed = 0
        skipped = 0
        total_prerequisites = 0

        for row in rows:
            course_id = row["course_id"]
            course_name = row["course_name"]
            summary_text = row["summary_text"]

            has_summary = summary_text and summary_text.strip()

            if has_summary:
                result = extract_prerequisites_for_course(
                    course_name, summary_text, model
                )
            else:
                logger.debug(
                    "Course %d ('%s') has no summary — using name-only extraction",
                    course_id,
                    course_name,
                )
                result = extract_prerequisites_from_name_only(course_name, model)

            if result is None:
                logger.warning(
                    "Skipping course %d ('%s'): extraction returned None",
                    course_id,
                    course_name,
                )
                skipped += 1
                continue

            # Delete any existing prerequisites for re-processing support
            conn.execute(
                "DELETE FROM course_prerequisites WHERE course_id = ?",
                (course_id,),
            )

            # Insert extracted prerequisites (including prereq_confidence)
            prereqs_data = [
                (
                    course_id,
                    prereq.prereq_name,
                    prereq.relevance_strength,
                    prereq.reason_short,
                    prereq.prereq_confidence,
                )
                for prereq in result.prerequisites
            ]

            if prereqs_data:
                conn.executemany(
                    "INSERT INTO course_prerequisites "
                    "(course_id, prereq_name, relevance_strength, reason_short, "
                    "prereq_confidence) "
                    "VALUES (?, ?, ?, ?, ?)",
                    prereqs_data,
                )

            # Mark course as prereqs extracted
            conn.execute(
                "UPDATE courses SET prereqs_extracted = 1 WHERE course_id = ?",
                (course_id,),
            )

            # Atomic per-course commit (prerequisites + tracking flag)
            conn.commit()

            num_prereqs = len(result.prerequisites)
            total_prerequisites += num_prereqs
            processed += 1
            extraction_type = "summary" if has_summary else "name-only"
            logger.info(
                "[%d/%d] Processed course %d ('%s'): %d prerequisites [%s]",
                processed + skipped,
                total_courses,
                course_id,
                course_name,
                num_prereqs,
                extraction_type,
            )

        return {
            "processed": processed,
            "skipped": skipped,
            "total_prerequisites": total_prerequisites,
            "total_courses": total_courses,
        }

    finally:
        conn.close()
