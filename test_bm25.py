#!/usr/bin/env python3
"""
Simple test script for BM25 retrieval functionality.
Tests that the BM25 retrieval can find courses based on skill gaps.
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.schemas import SkillGap, SkillGapReport
from retrieval.bm25_retrieval import retrieve_by_skills


def test_bm25_retrieval():
    """Test BM25 retrieval with sample skill gaps."""

    print("🧪 Testing BM25 Retrieval Implementation")
    print("=" * 50)

    # Create sample skill gaps
    gaps = [
        SkillGap(
            skill_name="Python",
            required_level="intermediate",
            current_level="beginner",
            gap_type="weak",
            priority=0.8
        ),
        SkillGap(
            skill_name="Machine Learning",
            required_level="advanced",
            current_level=None,
            gap_type="missing",
            priority=0.9
        )
    ]

    gap_report = SkillGapReport(gaps=gaps, total_required=2)

    print(f"📋 Testing with {len(gaps)} skill gaps:")
    for gap in gaps:
        print(f"  • {gap.skill_name} ({gap.gap_type}: {gap.current_level or 'none'} → {gap.required_level})")

    try:
        # Test BM25 retrieval
        print("\n🔍 Running BM25 retrieval...")
        results = retrieve_by_skills(
            gap_report=gap_report,
            db_path=None  # Use default path
        )

        print(f"✅ Found {len(results)} courses")

        if results:
            print("\n📚 Top 3 results:")
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. {result['course_name']}")
                print(f"     BM25 Score: {result['bm25_score']:.4f}")
                print(f"     Matched Skills: {result['matched_skills']}")
                print()

        # Check if results are properly formatted
        required_keys = ['course_id', 'course_name', 'summary_text', 'bm25_score', 'matched_skills']
        if results:
            first_result = results[0]
            missing_keys = [key for key in required_keys if key not in first_result]
            if missing_keys:
                print(f"❌ Missing keys in result: {missing_keys}")
                return False
            else:
                print("✅ Results properly formatted")

        print("\n🎉 BM25 retrieval test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during BM25 retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test."""
    success = test_bm25_retrieval()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
