import sys
from pathlib import Path

import pytest

# Ensure project src is importable when running tests from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.schemas import CandidateCourse, RetrievalResult, SkillGap, SkillGapReport, UserContext
from retrieval.hybrid import reciprocal_rank_fusion, retrieve_candidates, weighted_fusion


def _dummy_gap_report() -> SkillGapReport:
    return SkillGapReport(
        gaps=[
            SkillGap(
                skill_name="kubernetes",
                required_level="advanced",
                current_level=None,
                gap_type="missing",
                priority=1.0,
            )
        ],
        total_required=1,
    )


def _dummy_user_context() -> UserContext:
    return UserContext(
        portal_id=123,
        completed_course_ids=[],
        completed_course_names=[],
        inferred_skills=[],
    )


def test_reciprocal_rank_fusion_combines_lists():
    bm25 = [
        {"course_id": 1},  # rank 1
        {"course_id": 2},  # rank 2
    ]
    semantic = [
        {"course_id": 2},  # rank 1
        {"course_id": 3},  # rank 2
    ]

    scores = reciprocal_rank_fusion([bm25, semantic], k=60)

    # Course 2 appears in both lists; should have the highest score
    assert scores[2] > scores[1]
    assert scores[2] > scores[3]
    # All three course IDs should be present
    assert set(scores.keys()) == {1, 2, 3}


def test_weighted_fusion_normalizes_and_applies_weights():
    bm25_results = [
        {"course_id": 10, "bm25_score": 2.0},
        {"course_id": 11, "bm25_score": 1.0},
    ]
    semantic_results = [
        {"course_id": 11, "semantic_score": 0.5},
        {"course_id": 12, "semantic_score": 1.0},
    ]

    scores = weighted_fusion(bm25_results, semantic_results, bm25_weight=0.4, semantic_weight=0.6)

    # Normalization: course 10 BM25 score should be 1.0 * 0.4
    assert pytest.approx(scores[10], rel=1e-6) == 0.4
    # Course 11 combines normalized bm25 (0.5) and semantic (0.5) with weights
    expected_11 = 0.5 * 0.4 + 0.5 * 0.6
    assert pytest.approx(scores[11], rel=1e-6) == expected_11
    # Course 12 has only semantic contribution (1.0 normalized * 0.6)
    assert pytest.approx(scores[12], rel=1e-6) == 0.6


def test_weighted_fusion_handles_empty_inputs():
    # If inputs are empty or max score is zero, should return an empty dict
    assert weighted_fusion([], []) == {}
    assert weighted_fusion([{"course_id": 1, "bm25_score": 0}], []) == {}


def test_retrieve_candidates_merges_sources_and_limits(monkeypatch):
    gap_report = _dummy_gap_report()
    user_context = _dummy_user_context()

    # Fake retrieval legs
    bm25_results = [
        {
            "course_id": 1,
            "course_name": "BM25 Only",
            "summary_text": "bm25 summary",
            "bm25_score": 0.9,
            "matched_skills": ["kubernetes"],
        },
        {
            "course_id": 2,
            "course_name": "Both Legs",
            "summary_text": "both summary",
            "bm25_score": 0.8,
            "matched_skills": ["containers"],
        },
    ]

    semantic_results = [
        {
            "course_id": 2,
            "course_name": "Both Legs",
            "semantic_score": 0.95,
        },
        {
            "course_id": 3,
            "course_name": "Semantic Only",
            "semantic_score": 0.6,
        },
    ]

    # Monkeypatch retrieval functions to avoid external dependencies and imports
    import types

    bm25_stub = types.SimpleNamespace(
        retrieve_by_skills=lambda gap_report, db_path=None, exclude_course_ids=None: bm25_results
    )
    semantic_stub = types.SimpleNamespace(
        retrieve_by_embedding=lambda gap_report, n_results=None, chroma_path=None, exclude_course_ids=None: semantic_results
    )

    monkeypatch.setitem(sys.modules, "retrieval.bm25_retrieval", bm25_stub)
    monkeypatch.setitem(sys.modules, "retrieval.semantic_retrieval", semantic_stub)

    result: RetrievalResult = retrieve_candidates(
        gap_report=gap_report,
        user_context=user_context,
        db_path=None,
        chroma_path=None,
        max_candidates=3,
    )

    # Ensure ordering by RRF score: course 2 (both) should lead, then 1, then 3
    ids_in_order = [c.course_id for c in result.candidates]
    assert ids_in_order == [2, 1, 3]

    # Source attribution counts
    assert result.bm25_count == 1  # course 1
    assert result.semantic_count == 1  # course 3
    assert result.both_count == 1  # course 2

    # Metadata propagation
    c2 = result.candidates[0]
    assert c2.course_name == "Both Legs"
    assert c2.summary_text == "both summary"  # prefers BM25 metadata
    assert c2.matched_skills == ["containers"]

    c3 = result.candidates[-1]
    assert c3.summary_text is None
    assert c3.matched_skills == []


def test_retrieve_candidates_empty_returns_empty(monkeypatch):
    gap_report = _dummy_gap_report()
    user_context = _dummy_user_context()

    import types

    bm25_stub = types.SimpleNamespace(
        retrieve_by_skills=lambda gap_report, db_path=None, exclude_course_ids=None: []
    )
    semantic_stub = types.SimpleNamespace(
        retrieve_by_embedding=lambda gap_report, n_results=None, chroma_path=None, exclude_course_ids=None: []
    )

    monkeypatch.setitem(sys.modules, "retrieval.bm25_retrieval", bm25_stub)
    monkeypatch.setitem(sys.modules, "retrieval.semantic_retrieval", semantic_stub)

    result = retrieve_candidates(
        gap_report=gap_report,
        user_context=user_context,
        db_path=None,
        chroma_path=None,
    )

    assert result.candidates == []
    assert result.bm25_count == 0
    assert result.semantic_count == 0
    assert result.both_count == 0
