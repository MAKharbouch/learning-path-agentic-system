"""Unit tests for semantic retrieval module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project src is importable when running tests from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.schemas import SkillGap, SkillGapReport
from retrieval.semantic_retrieval import retrieve_by_embedding


def _make_gap_report(skill_names: list[str], levels: list[str] | None = None) -> SkillGapReport:
    """Helper to create a SkillGapReport with given skill names and levels."""
    if levels is None:
        levels = ["advanced"] * len(skill_names)
    gaps = [
        SkillGap(
            skill_name=name,
            required_level=level,
            current_level=None,
            gap_type="missing",
            priority=1.0,
        )
        for name, level in zip(skill_names, levels)
    ]
    return SkillGapReport(gaps=gaps, total_required=len(gaps))


class TestRetrieveByEmbedding:
    """Tests for retrieve_by_embedding function."""

    def test_empty_gap_report_returns_empty_list(self):
        """If no skill gaps are provided, should return empty list."""
        gap_report = SkillGapReport(gaps=[], total_required=0)
        result = retrieve_by_embedding(gap_report, chroma_path=":memory:")
        assert result == []

    @patch("vectorstore.chroma.get_collection")
    def test_returns_empty_when_collection_empty(self, mock_get_collection):
        """Should return empty list when ChromaDB collection is empty."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_get_collection.return_value = mock_collection

        gap_report = _make_gap_report(["python"])
        result = retrieve_by_embedding(gap_report, chroma_path=":memory:")

        assert result == []

    @patch("vectorstore.chroma.get_collection")
    def test_queries_chroma_with_correct_parameters(self, mock_get_collection):
        """Should query ChromaDB with correct query text and parameters."""
        # Setup mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_collection.query.return_value = {
            "ids": [["1", "2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"course_name": "Course 1"}, {"course_name": "Course 2"}]],
        }
        mock_get_collection.return_value = mock_collection

        gap_report = _make_gap_report(["kubernetes"], ["advanced"])
        result = retrieve_by_embedding(gap_report, n_results=5, chroma_path=":memory:")

        # Verify collection.query was called
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args
        # Check n_results parameter
        assert call_args.kwargs.get("n_results") == 5

    @patch("vectorstore.chroma.get_collection")
    def test_excludes_specified_course_ids(self, mock_get_collection):
        """Should exclude courses in exclude_course_ids list."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.query.return_value = {
            "ids": [["1", "2", "3"]],
            "distances": [[0.1, 0.2, 0.3]],
            "metadatas": [
                [
                    {"course_name": "Course 1"},
                    {"course_name": "Course 2"},
                    {"course_name": "Course 3"},
                ]
            ],
        }
        mock_get_collection.return_value = mock_collection

        gap_report = _make_gap_report(["python"])
        # Exclude course_id 1
        result = retrieve_by_embedding(
            gap_report, chroma_path=":memory:", exclude_course_ids=[1]
        )

        # Course 1 should be excluded
        course_ids = [r["course_id"] for r in result]
        assert 1 not in course_ids

    @patch("vectorstore.chroma.get_collection")
    def test_returns_correct_result_structure(self, mock_get_collection):
        """Should return results with correct keys."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "ids": [["42"]],
            "distances": [[0.5]],
            "metadatas": [[{"course_name": "Python Basics"}]],
        }
        mock_get_collection.return_value = mock_collection

        gap_report = _make_gap_report(["python"])
        result = retrieve_by_embedding(gap_report, chroma_path=":memory:")

        assert len(result) == 1
        item = result[0]
        assert "course_id" in item
        assert "course_name" in item
        assert "distance" in item
        assert "rank" in item
        assert item["course_id"] == 42

    @patch("vectorstore.chroma.get_collection")
    def test_builds_query_from_multiple_gaps(self, mock_get_collection):
        """Should build query text from all skill gaps."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "distances": [[0.1]],
            "metadatas": [[{"course_name": "Course"}]],
        }
        mock_get_collection.return_value = mock_collection

        gap_report = _make_gap_report(
            ["kubernetes", "docker"], ["advanced", "intermediate"]
        )
        retrieve_by_embedding(gap_report, chroma_path=":memory:")

        # Check that query was made
        call_args = mock_collection.query.call_args
        query_texts = call_args.kwargs.get("query_texts")
        assert query_texts is not None
        # Query should contain skill names and levels
        query = query_texts[0]
        assert "kubernetes" in query.lower()
        assert "docker" in query.lower()
        assert "advanced" in query.lower()
        assert "intermediate" in query.lower()

    @patch("vectorstore.chroma.get_collection")
    def test_caps_n_results_to_collection_size(self, mock_get_collection):
        """Should cap n_results to collection size."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5  # Only 5 items in collection
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "distances": [[0.1]],
            "metadatas": [[{"course_name": "Course"}]],
        }
        mock_get_collection.return_value = mock_collection

        gap_report = _make_gap_report(["python"])
        # Request 100 results but only 5 exist
        retrieve_by_embedding(gap_report, n_results=100, chroma_path=":memory:")

        # Should cap at 5
        call_args = mock_collection.query.call_args
        assert call_args.kwargs.get("n_results") == 5

    @patch("vectorstore.chroma.get_collection")
    def test_rank_reflects_original_order(self, mock_get_collection):
        """Rank should reflect the original ChromaDB ordering."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        # Chroma returns in order of relevance
        mock_collection.query.return_value = {
            "ids": [["1", "2", "3"]],
            "distances": [[0.1, 0.2, 0.3]],
            "metadatas": [
                [
                    {"course_name": "First"},
                    {"course_name": "Second"},
                    {"course_name": "Third"},
                ]
            ],
        }
        mock_get_collection.return_value = mock_collection

        gap_report = _make_gap_report(["python"])
        result = retrieve_by_embedding(gap_report, chroma_path=":memory:")

        assert result[0]["rank"] == 1
        assert result[1]["rank"] == 2
        assert result[2]["rank"] == 3

    @patch("vectorstore.chroma.get_collection")
    def test_handles_missing_metadata_fields(self, mock_get_collection):
        """Should handle missing optional metadata fields gracefully."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        # Return empty metadata
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "distances": [[0.1]],
            "metadatas": [[{}]],
        }
        mock_get_collection.return_value = mock_collection

        gap_report = _make_gap_report(["python"])
        result = retrieve_by_embedding(gap_report, chroma_path=":memory:")

        # Should still return course with empty name
        assert len(result) == 1
        assert result[0]["course_name"] == ""

    @patch("vectorstore.chroma.get_collection")
    def test_handles_empty_distances(self, mock_get_collection):
        """Should handle case when distances are empty."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "distances": [],  # Empty distances
            "metadatas": [[{"course_name": "Course"}]],
        }
        mock_get_collection.return_value = mock_collection

        gap_report = _make_gap_report(["python"])
        result = retrieve_by_embedding(gap_report, chroma_path=":memory:")

        # Empty distances results in no output (actual behavior)
        assert result == []

    @patch("vectorstore.chroma.get_collection")
    def test_handles_empty_metadatas(self, mock_get_collection):
        """Should handle case when metadatas are empty."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "distances": [[0.1]],
            "metadatas": [],  # Empty metadatas
        }
        mock_get_collection.return_value = mock_collection

        gap_report = _make_gap_report(["python"])
        result = retrieve_by_embedding(gap_report, chroma_path=":memory:")

        # Empty metadatas results in no output (actual behavior)
        assert result == []

    @patch("vectorstore.chroma.get_collection")
    def test_all_excluded_ids_removed(self, mock_get_collection):
        """Should remove all excluded IDs, not just some."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["1", "2", "3", "4", "5"]],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "metadatas": [
                [
                    {"course_name": "C1"},
                    {"course_name": "C2"},
                    {"course_name": "C3"},
                    {"course_name": "C4"},
                    {"course_name": "C5"},
                ]
            ],
        }
        mock_get_collection.return_value = mock_collection

        gap_report = _make_gap_report(["python"])
        # Exclude all courses
        result = retrieve_by_embedding(
            gap_report, chroma_path=":memory:", exclude_course_ids=[1, 2, 3, 4, 5]
        )

        assert result == []
