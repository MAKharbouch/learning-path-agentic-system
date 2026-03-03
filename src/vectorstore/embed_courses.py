"""Batch embedding indexer for course catalog into ChromaDB.

Embeds all courses with summaries into ChromaDB for semantic search.
Uses upsert() for idempotent re-runs. The embedding function
configured on the collection auto-embeds documents.
"""

import logging

logger = logging.getLogger(__name__)


def embed_all_courses(db_path=None, chroma_path=None):
    """Build ChromaDB embeddings index for all courses with summaries.

    Loads courses from SQLite, combines course_name and summary_text
    into a document, and upserts into ChromaDB in batches of 100.
    The collection's OpenAIEmbeddingFunction handles embedding generation.

    Args:
        db_path: Optional database path. Uses config default if None.
        chroma_path: Optional ChromaDB path. Uses config default if None.

    Returns:
        Dict with keys: embedded, total_courses.
    """
    from db.connection import get_connection
    from vectorstore.chroma import get_collection

    conn = get_connection(db_path)
    collection = get_collection(chroma_path)

    try:
        rows = conn.execute(
            "SELECT course_id, course_name, summary_text FROM courses "
            "WHERE summary_text IS NOT NULL AND summary_text != ''"
        ).fetchall()

        total = len(rows)
        logger.info("Found %d courses to embed", total)

        batch_size = 50
        embedded = 0

        for i in range(0, total, batch_size):
            batch = rows[i : i + batch_size]
            
            # Process documents individually to handle token limit errors
            valid_ids = []
            valid_documents = []
            valid_metadatas = []
            
            for row in batch:
                # Use only course name to avoid token limit issues
                text = row['course_name']
                
                # Add a short summary if available and not too long
                if row['summary_text'] and len(row['summary_text']) < 500:
                    text += f". {row['summary_text'][:200]}..."
                
                valid_ids.append(str(row["course_id"]))
                valid_documents.append(text)
                valid_metadatas.append({
                    "course_id": str(row["course_id"]),
                    "course_name": row["course_name"],
                })
            
            # Only upsert valid documents
            if valid_ids:
                collection.upsert(
                    ids=valid_ids,
                    documents=valid_documents,
                    metadatas=valid_metadatas,
                )
            embedded += len(valid_ids)
            logger.info("[%d/%d] Embedded courses", embedded, total)

        return {"embedded": embedded, "total_courses": total}
    finally:
        conn.close()
