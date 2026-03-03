"""BM25-based course retrieval by skill name matching.

Uses the BM25 algorithm to rank courses based on the relevance of their
skill descriptions to the user's skill gaps.
"""

import logging
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def retrieve_by_skills(
    gap_report,
    db_path: Path | None = None,
    exclude_course_ids: list[int] | None = None,
) -> list[dict]:
    """Retrieve courses whose skills match the skill gap names using BM25.

    Args:
        gap_report: A SkillGapReport with .gaps list (each gap has .skill_name).
        db_path: Path to SQLite database. If None, uses config default.
        exclude_course_ids: Course IDs to exclude (e.g., already completed).

    Returns:
        List of dicts with keys: course_id, course_name, summary_text,
        bm25_score, matched_skills.
    """
    from db.connection import get_connection

    skill_names = [gap.skill_name for gap in gap_report.gaps]
    if not skill_names:
        logger.info("No skill gaps provided, returning empty results")
        return []

    # Get all courses and their skills from database
    conn = get_connection(db_path)
    try:
        # Get all courses with their skills
        sql = """
            SELECT 
                cs.course_id,
                c.course_name,
                c.summary_text,
                GROUP_CONCAT(cs.skill_name) AS skills_text
            FROM course_skills cs
            JOIN courses c ON cs.course_id = c.course_id
            GROUP BY cs.course_id, c.course_name, c.summary_text
        """
        
        rows = conn.execute(sql).fetchall()
        
        if not rows:
            logger.info("No courses found in database")
            return []
        
        # Prepare documents for LangChain BM25Retriever
        documents = []
        course_data = []
        
        for row in rows:
            # Create Document with course skills as page_content
            skills_text = row['skills_text'] or ""
            doc = Document(
                page_content=skills_text,
                metadata={
                    'course_id': row['course_id'],
                    'course_name': row['course_name'],
                    'summary_text': row['summary_text'],
                    'skills_text': skills_text
                }
            )
            documents.append(doc)
            course_data.append(dict(row))
        
        # Create BM25 Retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        
        # Create query from skill names
        query = " ".join(skill_names)
        
        # Retrieve relevant documents
        retrieved_docs = bm25_retriever.invoke(query)
        
        # Prepare results with scores
        results = []
        exclude_set = set(exclude_course_ids) if exclude_course_ids else set()
        
        for doc in retrieved_docs:
            course_id = doc.metadata['course_id']
            if course_id in exclude_set:
                continue
                
            # Find matched skills
            course_skills = doc.metadata['skills_text'].split(',') if doc.metadata['skills_text'] else []
            matched_skills = []
            for skill in skill_names:
                for course_skill in course_skills:
                    if skill.lower().strip() == course_skill.lower().strip():
                        matched_skills.append(course_skill.strip())
                        break
            
            results.append({
                'course_id': course_id,
                'course_name': doc.metadata['course_name'],
                'summary_text': doc.metadata['summary_text'],
                'bm25_score': 1.0,  # LangChain BM25Retriever doesn't expose raw scores
                'matched_skills': matched_skills
            })
        
        # Sort by relevance (BM25Retriever already returns sorted results)
        logger.info("BM25 retrieval found %d courses for %d skill gaps", len(results), len(skill_names))
        return results
        
    finally:
        conn.close()
