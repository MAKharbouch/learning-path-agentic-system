"""Deterministic validation checks for generated learning plans.

All functions in this module are pure -- no LLM calls, no DB access,
no side effects. They take a LearningPlanResponse + context data and
return validation violations.

validate_plan() is the main entry point that runs all checks.
"""

import logging

from models.schemas import ValidationResult, ValidationViolation

logger = logging.getLogger(__name__)


def validate_plan(plan_response, prereqs_by_course, gap_report, budget_hours):
    """Run all validation checks on a learning plan.

    Args:
        plan_response: A LearningPlanResponse from the LLM.
        prereqs_by_course: Dict mapping course_id to list of prereq dicts,
            each with keys: prereq_name, relevance_strength, reason_short.
        gap_report: A SkillGapReport with gaps to cover.
        budget_hours: Max training hours (float), or None if unconstrained.

    Returns:
        ValidationResult with is_valid and list of violations.
    """
    if not plan_response.courses:
        return ValidationResult(is_valid=True)

    violations = []
    violations.extend(_check_prerequisite_ordering(plan_response, prereqs_by_course))
    violations.extend(_check_hour_budget(plan_response, budget_hours))
    violations.extend(_check_skill_coverage(plan_response, gap_report))
    violations.extend(_check_duplicates(plan_response))
    violations.extend(_check_course_order(plan_response))

    return ValidationResult(is_valid=len(violations) == 0, violations=violations)


def _check_prerequisite_ordering(plan_response, prereqs_by_course):
    """Check that prerequisite courses appear before courses that need them.

    For each course, looks at its prerequisites (relevance_strength >= 0.5)
    and verifies that an earlier course in the plan satisfies each prereq
    via bidirectional case-insensitive substring matching against course
    names and targeted skills.

    Args:
        plan_response: A LearningPlanResponse.
        prereqs_by_course: Dict mapping course_id to list of prereq dicts.

    Returns:
        List of ValidationViolation for unsatisfied prerequisites.
    """
    violations = []
    sorted_courses = sorted(plan_response.courses, key=lambda c: c.course_order)

    for idx, course in enumerate(sorted_courses):
        course_prereqs = prereqs_by_course.get(course.course_id, [])
        # Filter to relevant prereqs only
        relevant_prereqs = [
            p for p in course_prereqs if p.get("relevance_strength", 0) >= 0.5
        ]

        for prereq in relevant_prereqs:
            prereq_name = prereq["prereq_name"].lower().strip()
            satisfied = False

            # Check all earlier courses (lower course_order)
            for earlier in sorted_courses[:idx]:
                earlier_name = earlier.course_name.lower().strip()
                earlier_skills = [s.lower().strip() for s in earlier.targeted_skills]

                # Bidirectional substring matching against course name
                if prereq_name in earlier_name or earlier_name in prereq_name:
                    satisfied = True
                    break

                # Bidirectional substring matching against targeted skills
                for skill in earlier_skills:
                    if prereq_name in skill or skill in prereq_name:
                        satisfied = True
                        break
                if satisfied:
                    break

            if not satisfied:
                violations.append(
                    ValidationViolation(
                        check_name="prerequisite_ordering",
                        severity="error",
                        message=(
                            f"Course '{course.course_name}' (order {course.course_order}) "
                            f"requires prerequisite '{prereq['prereq_name']}' but no earlier "
                            f"course in the plan satisfies it."
                        ),
                        course_id=course.course_id,
                    )
                )

    return violations


def _check_hour_budget(plan_response, budget_hours):
    """Check that total plan hours do not exceed the budget.

    Also verifies consistency between declared total_estimated_hours
    and the sum of individual course durations.

    Args:
        plan_response: A LearningPlanResponse.
        budget_hours: Max training hours (float), or None if unconstrained.

    Returns:
        List of ValidationViolation for budget overruns or inconsistencies.
    """
    violations = []
    actual_hours = sum(c.estimated_duration_hours for c in plan_response.courses)

    # Check budget constraint
    if budget_hours is not None and budget_hours > 0:
        if actual_hours > budget_hours:
            overage = actual_hours - budget_hours
            violations.append(
                ValidationViolation(
                    check_name="hour_budget",
                    severity="error",
                    message=(
                        f"Total plan hours ({actual_hours:.1f}h) exceed budget "
                        f"({budget_hours:.1f}h) by {overage:.1f}h."
                    ),
                )
            )

    # Check hours consistency
    if abs(plan_response.total_estimated_hours - actual_hours) > 0.1:
        violations.append(
            ValidationViolation(
                check_name="hours_consistency",
                severity="warning",
                message=(
                    f"Declared total_estimated_hours ({plan_response.total_estimated_hours:.1f}h) "
                    f"differs from sum of course durations ({actual_hours:.1f}h)."
                ),
            )
        )

    return violations


def _check_skill_coverage(plan_response, gap_report):
    """Check that high-priority skill gaps are covered by the plan.

    For each gap with priority >= 0.5, verifies that at least one course
    in the plan addresses it via bidirectional case-insensitive substring
    matching against targeted_skills.

    Args:
        plan_response: A LearningPlanResponse.
        gap_report: A SkillGapReport with gaps to cover.

    Returns:
        List of ValidationViolation (warnings) for uncovered gaps.
    """
    violations = []

    # Collect all targeted skills from plan courses
    plan_skills = set()
    for course in plan_response.courses:
        for skill in course.targeted_skills:
            plan_skills.add(skill.lower().strip())

    # Check each high-priority gap
    for gap in gap_report.gaps:
        if gap.priority < 0.5:
            continue

        gap_name = gap.skill_name.lower().strip()
        covered = False

        for plan_skill in plan_skills:
            if gap_name in plan_skill or plan_skill in gap_name:
                covered = True
                break

        if not covered:
            violations.append(
                ValidationViolation(
                    check_name="skill_coverage",
                    severity="warning",
                    message=(
                        f"High-priority skill gap '{gap.skill_name}' "
                        f"(priority {gap.priority:.2f}) is not covered by any "
                        f"course in the plan."
                    ),
                )
            )

    return violations


def _check_duplicates(plan_response):
    """Check for duplicate course IDs in the plan.

    Args:
        plan_response: A LearningPlanResponse.

    Returns:
        List of ValidationViolation for duplicate course IDs.
    """
    violations = []
    seen = set()

    for course in plan_response.courses:
        if course.course_id in seen:
            violations.append(
                ValidationViolation(
                    check_name="duplicate_course",
                    severity="error",
                    message=(
                        f"Course ID {course.course_id} ('{course.course_name}') "
                        f"appears multiple times in the plan."
                    ),
                    course_id=course.course_id,
                )
            )
        seen.add(course.course_id)

    return violations


def _check_course_order(plan_response):
    """Check that course_order values are sequential starting from 1.

    Args:
        plan_response: A LearningPlanResponse.

    Returns:
        List of ValidationViolation for non-sequential ordering.
    """
    violations = []
    actual_orders = sorted(c.course_order for c in plan_response.courses)
    expected_orders = list(range(1, len(plan_response.courses) + 1))

    if actual_orders != expected_orders:
        violations.append(
            ValidationViolation(
                check_name="course_order",
                severity="error",
                message=(
                    f"Course order values {actual_orders} are not sequential. "
                    f"Expected {expected_orders}."
                ),
            )
        )

    return violations
