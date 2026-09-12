"""
Unit tests for the offline evaluation framework.
Verifies evaluation question schema, metric formulas, threshold adherence,
category counts, and strict isolation of synthetic ground-truth data without calling APIs.
"""

from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
import pytest

from app.evaluation.evaluation_dataset import (
    CATEGORY_CONTRADICTION_LIMITATION,
    CATEGORY_EVIDENCE_RETRIEVAL,
    CATEGORY_PRIORITIZATION,
    CATEGORY_PRODUCT_INSIGHT,
    CATEGORY_THEME_DISCOVERY,
    generate_evaluation_questions,
    save_evaluation_dataset,
)
from app.evaluation.metrics import (
    calculate_citation_correctness,
    calculate_decision_preservation,
    calculate_evidence_retrieval_precision,
    calculate_evidence_support_rate,
    calculate_important_theme_recall,
    calculate_insight_quality,
    calculate_pm_usefulness,
    run_full_evaluation,
)


@pytest.fixture
def sample_eval_inputs():
    """Create a self-contained evaluation fixture."""
    canonical_df = pd.DataFrame(
        {
            "feedback_id": [f"CAN-FB-{i:03d}" for i in range(30)],
            "theme": [
                "Payment & Checkout Reliability" if i < 10 else ("Product Quality Issues" if i < 20 else "Unknown")
                for i in range(30)
            ],
            "theme_origin": [
                "synthetic_ground_truth" if i < 20 else "not_available"
                for i in range(30)
            ],
        }
    )

    evidence_bundles = [
        {
            "cluster_id": 0,
            "theme_name": "Payment Status Ambiguity During Checkout",
            "problem_summary": "Payment timeout on mobile checkout",
            "supporting_feedback": [
                {"feedback_id": f"CAN-FB-{i:03d}", "similarity_score": 0.95 - (i * 0.01)}
                for i in range(5)
            ],
        },
        {
            "cluster_id": 1,
            "theme_name": "Poor Product Quality",
            "problem_summary": "Below standard build quality",
            "supporting_feedback": [
                {"feedback_id": f"CAN-FB-{i:03d}", "similarity_score": 0.95 - (i * 0.01)}
                for i in range(10, 15)
            ],
        },
    ]

    priority_assessments = [
        {"cluster_id": 0, "decision": "BUILD", "priority_score": 0.45},
        {"cluster_id": 1, "decision": "INVESTIGATE", "priority_score": 0.18},
    ]

    product_insights = [
        {
            "cluster_id": 0,
            "theme_name": "Payment Status Ambiguity During Checkout",
            "insight": "Customers experience payment gateway spinning and timeout delays.",
            "customer_problem": "Lack of real-time transaction confirmation.",
            "recommended_action": "Build payment polling status fallback.",
            "decision": "BUILD",
            "priority_score": 0.45,
            "supporting_feedback_ids": ["CAN-FB-000", "CAN-FB-001"],
            "confidence": 0.95,
            "contradiction_flag": False,
            "limitations": "Backend logs unavailable.",
        },
        {
            "cluster_id": 1,
            "theme_name": "Poor Product Quality",
            "insight": "Feedback reflects customer dissatisfaction with item build quality.",
            "customer_problem": "Substandard material quality.",
            "recommended_action": "Investigate supplier defect rates.",
            "decision": "INVESTIGATE",
            "priority_score": 0.18,
            "supporting_feedback_ids": ["CAN-FB-010", "CAN-FB-011"],
            "confidence": 0.85,
            "contradiction_flag": True,
            "limitations": "Limited sample size.",
        },
    ]

    return canonical_df, evidence_bundles, priority_assessments, product_insights


def test_evaluation_dataset_question_counts():
    """Verify exactly 30 evaluation questions with exact category distribution."""
    questions = generate_evaluation_questions()
    assert len(questions) == 30

    cat_counts = {}
    for q in questions:
        c = q["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    assert cat_counts[CATEGORY_EVIDENCE_RETRIEVAL] == 8
    assert cat_counts[CATEGORY_THEME_DISCOVERY] == 6
    assert cat_counts[CATEGORY_PRIORITIZATION] == 6
    assert cat_counts[CATEGORY_PRODUCT_INSIGHT] == 6
    assert cat_counts[CATEGORY_CONTRADICTION_LIMITATION] == 4


def test_evaluation_questions_schema():
    """Verify all questions adhere to required dictionary schema."""
    questions = generate_evaluation_questions()
    required_keys = {
        "question_id",
        "category",
        "question",
        "expected_evidence_ids",
        "expected_theme",
        "expected_decision",
        "evaluation_notes",
    }
    for q in questions:
        assert required_keys.issubset(set(q.keys()))
        assert len(q["question"]) > 10
        assert isinstance(q["expected_evidence_ids"], list)


def test_calculate_evidence_support_rate(sample_eval_inputs):
    """Verify Evidence Support Rate calculation."""
    _, _, _, insights = sample_eval_inputs
    rate, details = calculate_evidence_support_rate(insights)
    assert rate == 1.0
    assert details["supported_count"] == 2


def test_calculate_citation_correctness(sample_eval_inputs):
    """Verify Citation Correctness calculation."""
    canonical_df, bundles, _, insights = sample_eval_inputs
    canonical_fids = set(canonical_df["feedback_id"])

    rate, details = calculate_citation_correctness(insights, canonical_fids, bundles)
    assert rate == 1.0
    assert details["correct_citations"] == 4
    assert details["total_citations"] == 4


def test_calculate_evidence_retrieval_precision():
    """Verify precision matching between retrieved and expected evidence sets."""
    eval_q = [
        {
            "question_id": "EV-01",
            "category": "Evidence Retrieval",
            "expected_evidence_ids": ["FB-001", "FB-002", "FB-003"],
            "expected_theme": "Payment Ambiguity",
        }
    ]
    bundles = [
        {
            "cluster_id": 0,
            "theme_name": "Payment Ambiguity",
            "supporting_feedback": [
                {"feedback_id": "FB-001"},
                {"feedback_id": "FB-002"},
                {"feedback_id": "FB-099"},
            ],
        }
    ]

    precision, details = calculate_evidence_retrieval_precision(eval_q, bundles)
    # Overlap is 2 out of 3 -> 0.6667
    assert precision > 0.60
    assert details[0]["overlap_count"] == 2


def test_calculate_important_theme_recall(sample_eval_inputs):
    """Verify Important Theme Recall evaluates ONLY synthetic ground truth."""
    canonical_df, _, _, insights = sample_eval_inputs

    recall, details = calculate_important_theme_recall(canonical_df, insights)
    assert recall >= 0.70
    assert "Payment & Checkout Reliability" in details["matched_themes"]
    assert "Product Quality Issues" in details["matched_themes"]


def test_calculate_insight_quality(sample_eval_inputs):
    """Verify deterministic 8-point rubric for Insight Quality."""
    _, _, _, insights = sample_eval_inputs
    quality, details = calculate_insight_quality(insights)
    assert quality >= 0.85
    assert len(details) == 2


def test_calculate_pm_usefulness(sample_eval_inputs):
    """Verify deterministic 5-point proxy rubric for PM Usefulness."""
    _, _, _, insights = sample_eval_inputs
    usefulness, details = calculate_pm_usefulness(insights)
    assert usefulness >= 0.80
    assert len(details) == 2


def test_calculate_decision_preservation(sample_eval_inputs):
    """Verify Decision Preservation detects any alterations."""
    _, _, priority_assessments, insights = sample_eval_inputs
    preservation, details = calculate_decision_preservation(insights, priority_assessments)
    assert preservation == 1.0
    assert all(d["preserved"] for d in details)


def test_full_evaluation_execution(sample_eval_inputs):
    """Verify run_full_evaluation bundles all 7 metrics with pass/fail status."""
    canonical_df, bundles, priority_assessments, insights = sample_eval_inputs
    questions = generate_evaluation_questions()

    results = run_full_evaluation(
        canonical_df=canonical_df,
        evidence_bundles=bundles,
        priority_assessments=priority_assessments,
        product_insights=insights,
        evaluation_questions=questions,
    )

    metrics = results["metrics"]
    assert metrics["evidence_support_rate"] == 1.0
    assert metrics["decision_preservation"] == 1.0
    assert len(results["question_results"]) == 30


def test_save_evaluation_dataset():
    """Verify saving evaluation dataset to disk."""
    questions = generate_evaluation_questions()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_p = Path(tmpdir) / "eval_q.json"
        save_evaluation_dataset(questions, out_p)
        assert out_p.exists()
        with open(out_p, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert len(loaded) == 30
