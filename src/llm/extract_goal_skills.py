"""LLM-based goal skill extraction using structured output.

Extracts required skills from a free-text learning goal.
"""

import logging
from typing import cast

logger = logging.getLogger(__name__)


def extract_required_skills(goal_text):
    """Extract required skills from a learning goal using LLM structured output.

    Uses the configured LLM provider with structured output to parse a
    free-text learning goal into a list of RequiredSkill items.

    Args:
        goal_text: Free-text description of the learning goal.

    Returns:
        GoalSkillsResponse with 1-10 RequiredSkill items.
    """
    from langchain_core.prompts import ChatPromptTemplate

    from llm.client import get_chat_model
    from llm.prompts import GOAL_SKILL_EXTRACTION_PROMPT
    from models.schemas import GoalSkillsResponse

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GOAL_SKILL_EXTRACTION_PROMPT),
            ("user", "Learning goal: {goal_text}"),
        ]
    )

    model = get_chat_model()
    structured_model = model.with_structured_output(GoalSkillsResponse)

    chain = prompt | structured_model
    result = cast(GoalSkillsResponse, chain.invoke({"goal_text": goal_text}))

    logger.info("Extracted %d required skills from goal", len(result.skills))

    return result
