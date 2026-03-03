"""Semantic course retrieval via ChromaDB embedding similarity.

Queries the ChromaDB vector store to find courses whose embeddings
are closest to a concatenated skill gap query string.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def retrieve_by_embedding(
    gap_report,
    n_results: int = 20,
    chroma_path: Path | None = None,
    exclude_course_ids: list[int] | None = None,
) -> list[dict]:
    """Retrieve courses by embedding similarity to skill gaps.

    Args:
        gap_report: A SkillGapReport with .gaps list (each gap has
            .skill_name and .required_level).
        n_results: Maximum number of results to request from ChromaDB.
        chroma_path: Path to ChromaDB storage. If None, uses config default.
        exclude_course_ids: Course IDs to exclude (e.g., already completed).

    Returns:
        List of dicts with keys: course_id, course_name, distance, rank.
        Rank reflects the original ChromaDB ordering (before exclusion).
    """
    from vectorstore.chroma import get_collection

    if not gap_report.gaps:
        logger.info("No skill gaps provided, returning empty results")
        return []

    exclude_set = set(exclude_course_ids) if exclude_course_ids else set()

    # Build a single query string from all gaps
    parts = []
    for gap in gap_report.gaps:
        part = f"{gap.skill_name} {gap.required_level} level".strip()
        parts.append(part)
    query_text = ". ".join(parts)

    logger.info("Semantic query text length: %d chars", len(query_text))

    collection = get_collection(chroma_path)
    collection_size = collection.count()
    if collection_size == 0:
        logger.info("ChromaDB collection is empty, returning empty results")
        return []

    capped_n_results = min(n_results, collection_size)

    results = collection.query(
        query_texts=[query_text],
        n_results=capped_n_results,
        include=["metadatas", "documents", "distances"],
    )

    ids = results["ids"][0]
    distances = results["distances"][0] if results["distances"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    output = []
    for i, (doc_id, dist, meta) in enumerate(zip(ids, distances, metadatas)):
        course_id = int(doc_id)
        if course_id in exclude_set:
            continue
        output.append({
            "course_id": course_id,
            "course_name": meta.get("course_name", ""),
            "distance": dist,
            "rank": i + 1,
        })

    logger.info("Semantic retrieval found %d courses (after exclusion)", len(output))
    return output
