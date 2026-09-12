"""
Evaluation Dataset Generator for VoC Copilot Intelligence Pipeline.

Generates exactly 30 deterministic evaluation questions across 5 core categories
using the canonical dataset and authoritative pipeline outputs without calling external APIs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

EVALUATION_QUESTIONS_SEED = 42

CATEGORY_EVIDENCE_RETRIEVAL = "Evidence Retrieval"
CATEGORY_THEME_DISCOVERY = "Theme / Problem Discovery"
CATEGORY_PRIORITIZATION = "Prioritization / Decision"
CATEGORY_PRODUCT_INSIGHT = "Product Insight / Recommendation"
CATEGORY_CONTRADICTION_LIMITATION = "Contradiction / Limitation Awareness"


def generate_evaluation_questions() -> List[Dict[str, Any]]:
    """
    Generate 30 structured, deterministic evaluation questions across 5 categories.

    Category Distribution:
      A. Evidence Retrieval: 8 questions
      B. Theme / Problem Discovery: 6 questions
      C. Prioritization / Decision: 6 questions
      D. Product Insight / Recommendation: 6 questions
      E. Contradiction / Limitation Awareness: 4 questions
    """
    questions: List[Dict[str, Any]] = [
        # =========================================================================
        # Category A: Evidence Retrieval (8 questions) - 1 per canonical cluster
        # =========================================================================
        {
            "question_id": "EV-ER-01",
            "category": CATEGORY_EVIDENCE_RETRIEVAL,
            "question": "Which customer feedback records support checkout payment instability and session timeouts?",
            "expected_evidence_ids": [
                "SYN-INT-001", "SYN-ENH-001", "SYN-ENH-013", "SYN-ENH-017",
                "SYN-SUP-002", "SYN-INT-002", "SYN-SUP-001", "SYN-ENH-010"
            ],
            "expected_theme": "Checkout Payment Instability and Failure",
            "expected_decision": "BUILD",
            "evaluation_notes": "Evaluates precision of retrieved evidence for payment failure and redirect timeouts (Cluster 7).",
        },
        {
            "question_id": "EV-ER-02",
            "category": CATEGORY_EVIDENCE_RETRIEVAL,
            "question": "Which customer feedback records substantiate poor product quality and material breakdown?",
            "expected_evidence_ids": [
                "PUB-000257", "PUB-000347", "PUB-000057", "PUB-000209",
                "PUB-000147", "PUB-000301", "PUB-000172", "PUB-000060"
            ],
            "expected_theme": "Poor Product Quality and Durability",
            "expected_decision": "BUILD",
            "evaluation_notes": "Checks retrieval of customer complaints regarding substandard physical product quality (Cluster 4).",
        },
        {
            "question_id": "EV-ER-03",
            "category": CATEGORY_EVIDENCE_RETRIEVAL,
            "question": "Which customer feedback records provide evidence for mediocre product build and small component sizing?",
            "expected_evidence_ids": [
                "PUB-000140", "PUB-000298", "PUB-000104", "PUB-000143",
                "PUB-000238", "PUB-000051", "PUB-000263", "PUB-000340"
            ],
            "expected_theme": "Mediocre Product and Build Quality",
            "expected_decision": "INVESTIGATE",
            "evaluation_notes": "Evaluates evidence retrieval for lukewarm average product build reviews (Cluster 3).",
        },
        {
            "question_id": "EV-ER-04",
            "category": CATEGORY_EVIDENCE_RETRIEVAL,
            "question": "Which feedback records substantiate high overall customer praise and general product satisfaction?",
            "expected_evidence_ids": [
                "PUB-000223", "PUB-000315", "PUB-000331", "PUB-000335",
                "PUB-000027", "PUB-000253", "PUB-000164", "PUB-000247"
            ],
            "expected_theme": "High Overall Product Satisfaction",
            "expected_decision": "DON'T BUILD YET",
            "evaluation_notes": "Verifies retrieval of positive customer quality praise (Cluster 5).",
        },
        {
            "question_id": "EV-ER-05",
            "category": CATEGORY_EVIDENCE_RETRIEVAL,
            "question": "Which customer feedback records cite conflicting delivery timelines and date uncertainty?",
            "expected_evidence_ids": [
                "SYN-ENH-057", "SYN-SUR-010", "SYN-SUP-023", "SYN-SUP-032",
                "SYN-SUP-022", "SYN-ENH-078", "SYN-ENH-062", "SYN-SUP-027"
            ],
            "expected_theme": "Inaccurate and Inconsistent Delivery Estimates",
            "expected_decision": "BUILD",
            "evaluation_notes": "Checks retrieval of delivery date mismatch and carrier tracking records (Cluster 2).",
        },
        {
            "question_id": "EV-ER-06",
            "category": CATEGORY_EVIDENCE_RETRIEVAL,
            "question": "Which customer feedback records provide evidence for wishlist feature requests and AI assistant tools?",
            "expected_evidence_ids": [
                "SYN-ENH-151", "SYN-INT-029", "SYN-ENH-179", "SYN-ENH-162",
                "SYN-ENH-173", "SYN-SUR-023", "SYN-ENH-181", "SYN-ENH-144"
            ],
            "expected_theme": "Demand for AI Shopping Assistants",
            "expected_decision": "INVESTIGATE",
            "evaluation_notes": "Verifies retrieval of AI wishlist and feature enhancement suggestions (Cluster 0).",
        },
        {
            "question_id": "EV-ER-07",
            "category": CATEGORY_EVIDENCE_RETRIEVAL,
            "question": "Which records provide evidence of fast one-click checkout and seamless fulfillment experience?",
            "expected_evidence_ids": [
                "SYN-SUP-071", "SYN-ENH-265", "SYN-SUR-033", "SYN-SUP-074",
                "SYN-SUR-034", "SYN-ENH-299", "SYN-SUP-073", "SYN-ENH-286"
            ],
            "expected_theme": "Seamless Checkout and Delivery Experience",
            "expected_decision": "MONITOR",
            "evaluation_notes": "Checks retrieval of positive checkout and delivery feedback samples (Cluster 1).",
        },
        {
            "question_id": "EV-ER-08",
            "category": CATEGORY_EVIDENCE_RETRIEVAL,
            "question": "Which customer feedback records capture diverse friction across search relevance, account onboarding, and order links?",
            "expected_evidence_ids": [
                "SYN-SUR-014", "SYN-SUR-022", "SYN-ENH-084", "SYN-INT-016",
                "SYN-SUR-018", "SYN-ENH-125", "SYN-ENH-105", "SYN-INT-013"
            ],
            "expected_theme": "Mixed Customer Feedback",
            "expected_decision": "BUILD",
            "evaluation_notes": "Evaluates multi-topic feedback evidence clustered across search and onboarding (Cluster 6).",
        },

        # =========================================================================
        # Category B: Theme / Problem Discovery (6 questions)
        # =========================================================================
        {
            "question_id": "EV-TD-01",
            "category": CATEGORY_THEME_DISCOVERY,
            "question": "Does the pipeline discover an actionable theme representing Payment & Checkout Reliability?",
            "expected_evidence_ids": ["SYN-INT-001", "SYN-ENH-001"],
            "expected_theme": "Checkout Payment Instability and Failure",
            "expected_decision": "BUILD",
            "evaluation_notes": "Verifies discovery of the primary synthetic checkout reliability theme (Cluster 7).",
        },
        {
            "question_id": "EV-TD-02",
            "category": CATEGORY_THEME_DISCOVERY,
            "question": "Does the pipeline discover a theme capturing severe product physical quality deficiencies?",
            "expected_evidence_ids": ["PUB-000257", "PUB-000347"],
            "expected_theme": "Poor Product Quality and Durability",
            "expected_decision": "BUILD",
            "evaluation_notes": "Checks semantic theme identification for low product quality (Cluster 4).",
        },
        {
            "question_id": "EV-TD-03",
            "category": CATEGORY_THEME_DISCOVERY,
            "question": "Does the pipeline discover a theme reflecting inaccurate delivery date estimates?",
            "expected_evidence_ids": ["SYN-ENH-057", "SYN-SUR-010"],
            "expected_theme": "Inaccurate and Inconsistent Delivery Estimates",
            "expected_decision": "BUILD",
            "evaluation_notes": "Verifies identification of delivery estimation inaccuracy (Cluster 2).",
        },
        {
            "question_id": "EV-TD-04",
            "category": CATEGORY_THEME_DISCOVERY,
            "question": "Does the pipeline discover a theme reflecting mediocre or average build quality?",
            "expected_evidence_ids": ["PUB-000140", "PUB-000298"],
            "expected_theme": "Mediocre Product and Build Quality",
            "expected_decision": "INVESTIGATE",
            "evaluation_notes": "Verifies identification of borderline satisfaction and average quality (Cluster 3).",
        },
        {
            "question_id": "EV-TD-05",
            "category": CATEGORY_THEME_DISCOVERY,
            "question": "Does the pipeline isolate high-volume positive customer satisfaction feedback into a distinct cluster?",
            "expected_evidence_ids": ["PUB-000223", "PUB-000315"],
            "expected_theme": "High Overall Product Satisfaction",
            "expected_decision": "DON'T BUILD YET",
            "evaluation_notes": "Verifies segregation of positive review feedback (Cluster 5).",
        },
        {
            "question_id": "EV-TD-06",
            "category": CATEGORY_THEME_DISCOVERY,
            "question": "Does the pipeline identify wishlist and AI shopping feature requests within customer feedback?",
            "expected_evidence_ids": ["SYN-ENH-151", "SYN-INT-029"],
            "expected_theme": "Demand for AI Shopping Assistants",
            "expected_decision": "INVESTIGATE",
            "evaluation_notes": "Verifies discovery of AI shopping and wishlist feature suggestions (Cluster 0).",
        },

        # =========================================================================
        # Category C: Prioritization / Decision (6 questions)
        # =========================================================================
        {
            "question_id": "EV-PD-01",
            "category": CATEGORY_PRIORITIZATION,
            "question": "What is the highest priority theme recommended for immediate BUILD development?",
            "expected_evidence_ids": ["SYN-ENH-057", "SYN-SUR-010"],
            "expected_theme": "Inaccurate and Inconsistent Delivery Estimates",
            "expected_decision": "BUILD",
            "evaluation_notes": "Verifies that Delivery Estimates receives top priority score (0.7179) and BUILD classification.",
        },
        {
            "question_id": "EV-PD-02",
            "category": CATEGORY_PRIORITIZATION,
            "question": "Why is Poor Product Quality and Durability (Cluster 4) classified as BUILD?",
            "expected_evidence_ids": ["PUB-000257", "PUB-000347"],
            "expected_theme": "Poor Product Quality and Durability",
            "expected_decision": "BUILD",
            "evaluation_notes": "Verifies BUILD decision driven by high severity (4.51/5) and priority score (0.5515).",
        },
        {
            "question_id": "EV-PD-03",
            "category": CATEGORY_PRIORITIZATION,
            "question": "What decision is assigned to Demand for AI Shopping Assistants (Cluster 0)?",
            "expected_evidence_ids": ["SYN-ENH-151", "SYN-INT-029"],
            "expected_theme": "Demand for AI Shopping Assistants",
            "expected_decision": "INVESTIGATE",
            "evaluation_notes": "Verifies INVESTIGATE decision for priority score 0.2145.",
        },
        {
            "question_id": "EV-PD-04",
            "category": CATEGORY_PRIORITIZATION,
            "question": "What decision is assigned to Mediocre Product and Build Quality (Cluster 3)?",
            "expected_evidence_ids": ["PUB-000140", "PUB-000298"],
            "expected_theme": "Mediocre Product and Build Quality",
            "expected_decision": "INVESTIGATE",
            "evaluation_notes": "Verifies INVESTIGATE decision for priority score 0.2474.",
        },
        {
            "question_id": "EV-PD-05",
            "category": CATEGORY_PRIORITIZATION,
            "question": "Why is High Overall Product Satisfaction (Cluster 5) classified as DON'T BUILD YET despite having 136 records?",
            "expected_evidence_ids": ["PUB-000223", "PUB-000315"],
            "expected_theme": "High Overall Product Satisfaction",
            "expected_decision": "DON'T BUILD YET",
            "evaluation_notes": "Verifies low severity (1.35/5) and low priority score (0.0154) keep theme in DON'T BUILD YET.",
        },
        {
            "question_id": "EV-PD-06",
            "category": CATEGORY_PRIORITIZATION,
            "question": "Why is Seamless Checkout and Delivery Experience (Cluster 1) classified as MONITOR?",
            "expected_evidence_ids": ["SYN-SUP-071", "SYN-ENH-265"],
            "expected_theme": "Seamless Checkout and Delivery Experience",
            "expected_decision": "MONITOR",
            "evaluation_notes": "Verifies MONITOR classification for priority score 0.1103.",
        },

        # =========================================================================
        # Category D: Product Insight / Recommendation (6 questions)
        # =========================================================================
        {
            "question_id": "EV-PI-01",
            "category": CATEGORY_PRODUCT_INSIGHT,
            "question": "What engineering action is recommended for payment checkout instability (Cluster 7)?",
            "expected_evidence_ids": ["SYN-INT-001", "SYN-ENH-001"],
            "expected_theme": "Checkout Payment Instability and Failure",
            "expected_decision": "BUILD",
            "evaluation_notes": "Checks recommendation for payment gateway retries and transaction status polling.",
        },
        {
            "question_id": "EV-PI-02",
            "category": CATEGORY_PRODUCT_INSIGHT,
            "question": "What product action is recommended for delivery estimate inaccuracies (Cluster 2)?",
            "expected_evidence_ids": ["SYN-ENH-057", "SYN-SUR-010"],
            "expected_theme": "Inaccurate and Inconsistent Delivery Estimates",
            "expected_decision": "BUILD",
            "evaluation_notes": "Verifies recommendation for unified ETA engine across product, checkout, and tracking.",
        },
        {
            "question_id": "EV-PI-03",
            "category": CATEGORY_PRODUCT_INSIGHT,
            "question": "What exploratory action is recommended for AI shopping assistants (Cluster 0)?",
            "expected_evidence_ids": ["SYN-ENH-151", "SYN-INT-029"],
            "expected_theme": "Demand for AI Shopping Assistants",
            "expected_decision": "INVESTIGATE",
            "evaluation_notes": "Verifies discovery spike and user testing recommendation for side-by-side comparison.",
        },
        {
            "question_id": "EV-PI-04",
            "category": CATEGORY_PRODUCT_INSIGHT,
            "question": "What monitoring action is recommended for high overall product satisfaction (Cluster 5)?",
            "expected_evidence_ids": ["PUB-000223", "PUB-000315"],
            "expected_theme": "High Overall Product Satisfaction",
            "expected_decision": "DON'T BUILD YET",
            "evaluation_notes": "Verifies recommendation to defer development due to absence of friction.",
        },
        {
            "question_id": "EV-PI-05",
            "category": CATEGORY_PRODUCT_INSIGHT,
            "question": "What recommendation is given for the multi-topic Mixed Customer Feedback cluster (Cluster 6)?",
            "expected_evidence_ids": ["SYN-SUR-014", "SYN-SUR-022"],
            "expected_theme": "Mixed Customer Feedback",
            "expected_decision": "BUILD",
            "evaluation_notes": "Verifies recommendation to partition cluster into dedicated search, onboarding, and tracking tracks.",
        },
        {
            "question_id": "EV-PI-06",
            "category": CATEGORY_PRODUCT_INSIGHT,
            "question": "What telemetry action is recommended for seamless checkout and delivery (Cluster 1)?",
            "expected_evidence_ids": ["SYN-SUP-071", "SYN-ENH-265"],
            "expected_theme": "Seamless Checkout and Delivery Experience",
            "expected_decision": "MONITOR",
            "evaluation_notes": "Verifies recommendation to set SLA alerts and baseline monitoring.",
        },

        # =========================================================================
        # Category E: Contradiction / Limitation Awareness (4 questions)
        # =========================================================================
        {
            "question_id": "EV-CL-01",
            "category": CATEGORY_CONTRADICTION_LIMITATION,
            "question": "Which theme clusters exhibit significant sentiment contradictions between positive and negative feedback?",
            "expected_evidence_ids": ["SYN-ENH-151", "PUB-000140", "PUB-000257"],
            "expected_theme": "Contradiction Detection",
            "expected_decision": "N/A",
            "evaluation_notes": "Verifies contradiction flags on Clusters 0, 3, and 4 (mixed sentiment items).",
        },
        {
            "question_id": "EV-CL-02",
            "category": CATEGORY_CONTRADICTION_LIMITATION,
            "question": "Which theme clusters have high severity but suffer from unknown customer segment limitations?",
            "expected_evidence_ids": ["PUB-000257", "PUB-000140"],
            "expected_theme": "Customer Segment Limitations",
            "expected_decision": "N/A",
            "evaluation_notes": "Identifies Cluster 4 (75% unknown) and Cluster 3 (93.2% unknown) as segment-limited.",
        },
        {
            "question_id": "EV-CL-03",
            "category": CATEGORY_CONTRADICTION_LIMITATION,
            "question": "How does the insight layer handle the lack of backend technical error logs for payment timeouts?",
            "expected_evidence_ids": ["SYN-INT-001", "SYN-ENH-001"],
            "expected_theme": "Checkout Payment Instability and Failure",
            "expected_decision": "BUILD",
            "evaluation_notes": "Verifies limitation note regarding lack of backend partner timeout error codes.",
        },
        {
            "question_id": "EV-CL-04",
            "category": CATEGORY_CONTRADICTION_LIMITATION,
            "question": "Why does Mixed Customer Feedback (Cluster 6) note theme divergence across search, onboarding, and tracking?",
            "expected_evidence_ids": ["SYN-SUR-014", "SYN-SUR-022"],
            "expected_theme": "Mixed Customer Feedback",
            "expected_decision": "BUILD",
            "evaluation_notes": "Verifies limitation note on divergent topics (search relevance vs. account setup friction).",
        },
    ]

    return questions


def save_evaluation_dataset(
    questions: List[Dict[str, Any]],
    output_path: Union[str, Path] = "data/evaluation/evaluation_questions.json",
) -> None:
    """Save evaluation dataset JSON to disk."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
