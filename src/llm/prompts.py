"""System prompts for LLM-based extraction tasks."""

SKILL_EXTRACTION_PROMPT = """\
You are an expert learning analyst. Given a course name and summary, extract \
the skills taught in the course.

Instructions:
- Extract between 1 and 5 skills per course.
- Only extract skills that are clearly taught or practiced in the course.
- If the summary is too vague or generic to extract meaningful skills, return \
an empty skills list.

For each skill, provide:
- skill_name: A concise, specific skill name (e.g., "Python programming", \
"data visualization", "project management").
- skill_level: One of "beginner", "intermediate", or "advanced", based on the \
depth described in the summary.
- skill_confidence: A float between 0.0 and 1.0 indicating how confident you \
are that this skill is genuinely taught in the course (1.0 = certain, \
0.5 = moderate confidence, below 0.3 = uncertain).
- topic_name: The broader topic area this skill belongs to (e.g., "Software \
Engineering", "Data Science", "Leadership").
- topic_weight: A float between 0.0 and 1.0 indicating how much of the course \
focuses on this topic (all topic_weights for a course should roughly sum to 1.0).
"""

PREREQUISITE_EXTRACTION_PROMPT = """\
You are an expert learning analyst. Given a course name and summary, extract \
the prerequisites (prior knowledge or skills) that a learner should have \
before taking this course.

Instructions:
- Extract between 0 and 5 prerequisites per course.
- Only extract prerequisites that are clearly implied by the course content.
- If the course appears to be introductory or has no prerequisites, return \
an empty prerequisites list.

For each prerequisite, provide:
- prereq_name: A concise, specific prerequisite name (e.g., "Python \
programming basics", "linear algebra", "project management fundamentals").
- relevance_strength: A float between 0.0 and 1.0 indicating how strongly \
this prerequisite is needed (1.0 = essential, 0.5 = helpful but not \
required, below 0.3 = nice to have).
- reason_short: A brief explanation of why this prerequisite is needed \
(max 1-2 sentences).
- prereq_confidence: A float between 0.0 and 1.0 indicating how confident \
you are that this is a genuine prerequisite (1.0 = certain, 0.5 = moderate \
confidence, below 0.3 = uncertain).
"""

NAME_ONLY_SKILL_EXTRACTION_PROMPT = """\
You are an expert learning analyst. You have ONLY a course name — no summary \
is available. Extract the most likely skills this course teaches based on the \
title alone.

IMPORTANT — Quality over Quantity:
- Only extract skills where the title gives clear, specific evidence.
- Prefer 1-3 well-founded skills over 5 speculative ones.
- Set skill_confidence at 0.6 or below; never exceed 0.7 without a summary.
- If the course name is too generic or ambiguous, return an empty skills list.
- Do NOT hallucinate skills the title does not directly imply.

For each skill, provide:
- skill_name: A concise, specific skill name.
- skill_level: One of "beginner", "intermediate", or "advanced".
- skill_confidence: A float between 0.0 and 0.7 (name-only cap).
- topic_name: The broader topic area this skill belongs to.
- topic_weight: A float between 0.0 and 1.0 (all weights should sum to ~1.0).
"""

NAME_ONLY_PREREQUISITE_EXTRACTION_PROMPT = """\
You are an expert learning analyst. You have ONLY a course name — no summary \
is available. Infer the most likely prerequisites based on the title alone.

IMPORTANT — Quality over Quantity:
- Only extract prerequisites where the title strongly implies them.
- Prefer 0-2 well-founded prerequisites over speculative ones.
- Set prereq_confidence at 0.5 or below; never exceed 0.6 without a summary.
- If the name suggests an introductory course, return an empty list.
- Do NOT hallucinate prerequisites the title does not directly imply.

For each prerequisite, provide:
- prereq_name: A concise, specific prerequisite name.
- relevance_strength: A float between 0.0 and 1.0.
- reason_short: A brief explanation (max 1-2 sentences).
- prereq_confidence: A float between 0.0 and 0.6 (name-only cap).
"""

GOAL_SKILL_EXTRACTION_PROMPT = """\
You are an expert learning analyst. Given a learning goal, extract the skills \
that a learner needs to achieve that goal.

Instructions:
- Extract between 1 and 10 skills that the learning goal requires.
- Focus on concrete, teachable skills (not soft traits like "motivation").
- Use clear, specific skill names (e.g., "Python programming", \
"data visualization", "project management").

For each skill, provide:
- skill_name: A concise, specific skill name.
- skill_level: One of "beginner", "intermediate", or "advanced", based on the \
depth implied by the goal.
- importance: A float between 0.0 and 1.0 indicating how critical this skill \
is for achieving the goal (1.0 = essential, 0.5 = helpful, below 0.3 = nice \
to have).
"""

PLAN_GENERATION_PROMPT = """\
You are an expert learning path designer. Given a learner's skill gaps and a \
list of candidate courses, create an optimized learning plan.

CONSTRAINTS:
- Select courses ONLY from the provided candidate list. Use EXACT course_id values.
- Total estimated hours MUST NOT exceed the training hour budget ({budget_hours} hours). \
If budget is "unconstrained", select the optimal set without hour limit.
- Order courses so prerequisites are completed before courses that need them.
- Maximize coverage of the skill gaps listed below.

SKILL GAPS (prioritized, highest first):
{skill_gaps}

CANDIDATE COURSES (with prerequisites and matched skills):
{candidates}

INSTRUCTIONS:
1. Select courses that best cover the skill gaps. Prefer courses matching \
multiple gaps and courses with higher retrieval scores.
2. For each selected course, assign a phase label: "foundation" for \
prerequisite/introductory courses, "core" for main skill-building courses, \
"advanced" for deepening or specialization courses.
3. Order courses with course_order starting at 1. Foundation courses first, \
then core, then advanced. Within a phase, order by prerequisite dependencies.
4. Estimate duration in hours for each course (use summary context clues; \
typical online courses are 2-8 hours, workshops 4-16 hours).
5. For targeted_skills, list the specific skill gaps this course addresses.
6. For targeted_topic, use the primary topic area of the course.
7. For targeted_level, use one of: "beginner", "intermediate", "advanced".
8. Write a brief reasoning_summary (1-2 sentences) explaining why this course \
was selected and its position in the plan.
9. Provide a skill_coverage_summary describing overall gap coverage and any \
gaps that could not be addressed by the available candidates.

If no candidates meaningfully address the skill gaps, return an empty courses list \
with an explanatory skill_coverage_summary.\
"""

PLAN_REPAIR_PROMPT = """\
You are an expert learning path architect. A previously generated learning \
plan has failed validation. Your task is to repair it.

VIOLATIONS FOUND:
{violations}

CURRENT PLAN:
{current_plan}

ORIGINAL CONSTRAINTS:
- Training hour budget: {budget_hours} hours. If "unconstrained", no hour limit.
- Skill gaps to cover (prioritized):
{skill_gaps}

CANDIDATE COURSES (you may only select from these):
{candidates}

INSTRUCTIONS:
1. Address EACH violation listed above. For prerequisite ordering violations, \
reorder courses so prerequisites come first. For hour budget violations, \
remove lower-priority courses to fit within budget. For coverage warnings, \
add courses from the candidate list if possible within budget.
2. Use ONLY courses from the candidate list. Use exact course_id values.
3. Maintain sequential course_order starting from 1.
4. Keep phase labels appropriate: "foundation" first, then "core", then "advanced".
5. Recalculate total_estimated_hours as the sum of all course durations.
6. Update skill_coverage_summary to reflect the repaired plan's coverage.\
"""
