"""Hybrid course retrieval combining BM25 and semantic results via RRF.

Reciprocal Rank Fusion merges ranked lists from BM25 skill-matching
and ChromaDB semantic search into a single ranked candidate list.
retrieve_candidates() is the Phase 6 entry point for downstream phases.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> dict[int, float]:
    """Merge multiple ranked lists into a single score dict using RRF.

    For each item in each ranked list, compute a contribution of
    1 / (k + rank) where rank is 1-based. Scores are accumulated
    per course_id across all lists.

    Args:
        ranked_lists: List of ranked lists. Each inner list contains dicts
            with at least a ``course_id`` key, ordered by relevance
            (index 0 = most relevant).
        k: Smoothing constant from the original RRF paper (default 60).

    Returns:
        Dict mapping course_id to accumulated RRF score.
    """
    scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank_0, item in enumerate(ranked_list):
            cid = item["course_id"]
            rank = rank_0 + 1  # 1-based
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def weighted_fusion(
    bm25_results: list[dict],
    semantic_results: list[dict],
    bm25_weight: float = 0.3,
    semantic_weight: float = 0.7,
) -> dict[int, float]:
    """Combine BM25 and semantic results using weighted scores.

    Args:
        bm25_results: List of BM25 results with course_id and bm25_score
        semantic_results: List of semantic results with course_id and semantic_score
        bm25_weight: Weight for BM25 scores (default 0.3)
        semantic_weight: Weight for semantic scores (default 0.7)

    Returns:
        Dict mapping course_id to weighted fusion score.
    """
    scores: dict[int, float] = {}
    
    # Normalize BM25 scores to 0-1 range
    if bm25_results:
        max_bm25 = max(item.get("bm25_score", 0) for item in bm25_results)
        if max_bm25 > 0:
            for item in bm25_results:
                cid = item["course_id"]
                normalized_bm25 = item.get("bm25_score", 0) / max_bm25
                scores[cid] = normalized_bm25 * bm25_weight
    
    # Normalize semantic scores to 0-1 range
    if semantic_results:
        max_semantic = max(item.get("semantic_score", 0) for item in semantic_results)
        if max_semantic > 0:
            for item in semantic_results:
                cid = item["course_id"]
                normalized_semantic = item.get("semantic_score", 0) / max_semantic
                scores[cid] = scores.get(cid, 0) + normalized_semantic * semantic_weight
    
    return scores


def retrieve_candidates(
    gap_report,
    user_context,
    db_path: Path | None = None,
    chroma_path: Path | None = None,
    max_candidates: int = 30,
):
    """Orchestrate hybrid retrieval: BM25 + semantic legs merged via RRF.

    This is the Phase 6 entry point that downstream phases call.
    It runs both retrieval legs, merges via Reciprocal Rank Fusion,
    builds CandidateCourse objects with source attribution, and returns
    a RetrievalResult.

    Args:
        gap_report: A SkillGapReport with .gaps list.
        user_context: A UserContext with .completed_course_ids.
        db_path: Path to SQLite database. If None, uses config default.
        chroma_path: Path to ChromaDB storage. If None, uses config default.
        max_candidates: Maximum number of candidates to return.

    Returns:
        A RetrievalResult with ranked candidates and source counts.
    """
    from models.schemas import CandidateCourse, RetrievalResult
    from retrieval.semantic_retrieval import retrieve_by_embedding
    from retrieval.bm25_retrieval import retrieve_by_skills

    exclude_ids = set(user_context.completed_course_ids)

    # Run both retrieval legs
    bm25_results = retrieve_by_skills(
        gap_report, db_path, list(exclude_ids) if exclude_ids else None
    )
    semantic_results = retrieve_by_embedding(
        gap_report,
        n_results=max_candidates,
        chroma_path=chroma_path,
        exclude_course_ids=list(exclude_ids) if exclude_ids else None,
    )

    logger.info(
        "Retrieval legs: %d BM25 results, %d semantic results",
        len(bm25_results),
        len(semantic_results),
    )

    # Handle empty case
    if not bm25_results and not semantic_results:
        logger.info("Both retrieval legs returned empty, returning empty result")
        return RetrievalResult(
            candidates=[], bm25_count=0, semantic_count=0, both_count=0
        )

    # Compute RRF scores
    rrf_scores = reciprocal_rank_fusion([bm25_results, semantic_results])
    logger.info("RRF merge produced %d unique courses", len(rrf_scores))

    # Build metadata lookups and source tracking sets
    bm25_meta: dict[int, dict] = {}
    bm25_ids: set[int] = set()
    for item in bm25_results:
        cid = item["course_id"]
        bm25_meta[cid] = item
        bm25_ids.add(cid)

    semantic_meta: dict[int, dict] = {}
    semantic_ids: set[int] = set()
    for item in semantic_results:
        cid = item["course_id"]
        semantic_meta[cid] = item
        semantic_ids.add(cid)

    # Build CandidateCourse objects
    candidates: list[CandidateCourse] = []
    for cid, score in rrf_scores.items():
        # Determine source attribution
        in_bm25 = cid in bm25_ids
        in_semantic = cid in semantic_ids
        if in_bm25 and in_semantic:
            source = "both"
        elif in_bm25:
            source = "bm25"
        else:
            source = "semantic"

        # Prefer BM25 metadata (has JOIN data with summary_text, matched_skills)
        bm25_item = bm25_meta.get(cid)
        sem_item = semantic_meta.get(cid)

        course_name = ""
        if bm25_item:
            course_name = bm25_item.get("course_name", "")
        elif sem_item:
            course_name = sem_item.get("course_name", "")

        # Parse matched_skills from BM25 result
        matched_skills: list[str] = []
        if bm25_item and bm25_item.get("matched_skills"):
            matched_skills = bm25_item["matched_skills"]

        summary_text = None
        if bm25_item and bm25_item.get("summary_text"):
            summary_text = bm25_item["summary_text"]

        candidates.append(
            CandidateCourse(
                course_id=cid,
                course_name=course_name,
                score=score,
                source=source,
                matched_skills=matched_skills,
                summary_text=summary_text,
            )
        )

    # Sort by RRF score descending, cap at max_candidates
    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[:max_candidates]

    # Count source distribution
    bm25_count = sum(1 for c in candidates if c.source == "bm25")
    semantic_count = sum(1 for c in candidates if c.source == "semantic")
    both_count = sum(1 for c in candidates if c.source == "both")

    logger.info(
        "Final candidates: %d (bm25=%d, semantic=%d, both=%d)",
        len(candidates),
        bm25_count,
        semantic_count,
        both_count,
    )

    return RetrievalResult(
        candidates=candidates,
        bm25_count=bm25_count,
        semantic_count=semantic_count,
        both_count=both_count,
    )
