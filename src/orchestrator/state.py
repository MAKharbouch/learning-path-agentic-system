"""TypedDict state definitions for the LangGraph pipeline.

PipelineState carries all data between nodes (total=False so every
field is optional -- nodes populate their slice incrementally).
InputState and OutputState define clean graph boundaries.
"""

from typing import Any

from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    """Full mutable state passed between LangGraph nodes.

    Fields are grouped by the pipeline phase that produces them.
    All Pydantic model fields use ``Any`` because Pydantic instances
    survive intact inside TypedDict state.
    """

    # Inputs (set by the graph invocation)
    portal_id: int
    goal_text: str

    # Phase 5 -- skill gap analysis
    user_context: Any  # UserContext
    goal_skills: Any  # GoalSkillsResponse
    gap_report: Any  # SkillGapReport

    # Phase 6 -- hybrid course retrieval
    retrieval_result: Any  # RetrievalResult

    # Phase 7 -- learning plan generation
    path_run_id: str
    plan_response: Any  # LearningPlanResponse

    # Shared context (loaded once in generate, reused by validate/repair)
    prereqs_by_course: Any  # dict[int, list[dict]]

    # Phase 8 -- validation and repair
    validation_result: Any  # ValidationResult
    repair_count: int

    # Error tracking
    error: str
    error_node: str


class InputState(TypedDict):
    """Public input schema for the LangGraph graph.

    total=True (default) -- both fields are required.
    """

    portal_id: int
    goal_text: str


class OutputState(TypedDict, total=False):
    """Public output schema for the LangGraph graph.

    total=False -- only populated fields are returned.
    """

    plan_response: Any  # LearningPlanResponse
    validation_result: Any  # ValidationResult
    path_run_id: str
    error: str
    error_node: str
