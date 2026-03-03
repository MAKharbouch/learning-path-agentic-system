"""Skill gap analysis: compare user skills against goal requirements.

Contains the pure computation engine and the full pipeline entry point.
"""

import logging

logger = logging.getLogger(__name__)

LEVEL_ORDER = {"beginner": 1, "intermediate": 2, "advanced": 3}


def compute_skill_gap(user_skills, required_skills):
    """Compute skill gaps between user skills and goal requirements.

    Pure function -- no LLM, no DB, no side effects.

    Args:
        user_skills: List of UserSkillInferred from user context.
        required_skills: List of RequiredSkill from goal extraction.

    Returns:
        SkillGapReport with gaps sorted by priority (desc), missing before weak.
    """
    from models.schemas import SkillGap, SkillGapReport

    user_skill_map = {s.skill_name.lower().strip(): s for s in user_skills}

    gaps = []
    for req in required_skills:
        req_key = req.skill_name.lower().strip()
        user_skill = user_skill_map.get(req_key)

        if user_skill is None:
            gaps.append(
                SkillGap(
                    skill_name=req.skill_name,
                    required_level=req.skill_level,
                    current_level=None,
                    gap_type="missing",
                    priority=req.importance,
                )
            )
        elif LEVEL_ORDER.get(user_skill.skill_level, 0) < LEVEL_ORDER.get(
            req.skill_level, 0
        ):
            gaps.append(
                SkillGap(
                    skill_name=req.skill_name,
                    required_level=req.skill_level,
                    current_level=user_skill.skill_level,
                    gap_type="weak",
                    priority=req.importance,
                )
            )
        # If user level >= required level: no gap, skip

    # Sort: highest priority first, missing before weak at same priority
    gaps.sort(key=lambda g: (-g.priority, g.gap_type != "missing"))

    report = SkillGapReport(gaps=gaps, total_required=len(required_skills))

    logger.info(
        "Computed %d gaps from %d required skills (%d missing, %d weak)",
        len(gaps),
        len(required_skills),
        sum(1 for g in gaps if g.gap_type == "missing"),
        sum(1 for g in gaps if g.gap_type == "weak"),
    )

    return report


def analyze_skill_gap(portal_id, goal_text, db_path=None):
    """Full Phase 5 pipeline entry point: user context + goal extraction + gap computation.

    Wires together SQL user context loading, LLM goal skill extraction, and
    deterministic gap analysis into a single call. This function will become
    a LangGraph node in Phase 9.

    Args:
        portal_id: The user's portal ID.
        goal_text: Free-text description of the learning goal.
        db_path: Optional database path. Uses config default if None.

    Returns:
        Tuple of (UserContext, GoalSkillsResponse, SkillGapReport).
    """
    from db.user_context import get_user_context
    from llm.extract_goal_skills import extract_required_skills

    # Step 1: Load user context (SQL, no LLM)
    user_context = get_user_context(portal_id, db_path)

    # Step 2: Extract required skills from goal (LLM call)
    goal_skills = extract_required_skills(goal_text)

    # Step 3: Compute skill gaps (pure Python)
    gap_report = compute_skill_gap(user_context.inferred_skills, goal_skills.skills)

    logger.info(
        "Skill gap analysis complete for portal_id %s: %d gaps",
        portal_id,
        len(gap_report.gaps),
    )

    return (user_context, goal_skills, gap_report)
