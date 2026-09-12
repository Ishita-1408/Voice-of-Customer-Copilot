"""
Tests for Streamlit MVP Dashboard Data Integrity, Consistency & Rendering Logic.
"""

from pathlib import Path
import pandas as pd
import pytest

from app.streamlit_app import load_data, render_decision_badge, format_recommended_action, DECISION_COLORS


def test_streamlit_load_data_integrity():
    """Verify load_data loads all 6 authoritative artifacts with correct shapes."""
    df_voc, df_clusters, theme_defs, bundles, priority_assessments, product_insights = load_data()

    assert len(df_voc) == 800
    assert len(df_clusters) == 800
    assert len(theme_defs) == 8
    assert len(bundles) == 8
    assert len(priority_assessments) == 8
    assert len(product_insights) == 8


def test_streamlit_kpi_metrics():
    """Verify KPI metrics match exact expectations (800 records, 8 themes, 4 BUILD, 2 INVESTIGATE, 1 MONITOR, 1 DON'T BUILD YET)."""
    _, _, _, _, priority_assessments, _ = load_data()

    decisions = [p["decision"] for p in priority_assessments]
    assert len(decisions) == 8
    assert decisions.count("BUILD") == 4
    assert decisions.count("INVESTIGATE") == 2
    assert decisions.count("MONITOR") == 1
    assert decisions.count("DON'T BUILD YET") == 1


def test_streamlit_evidence_traceability_resolution():
    """Verify every cited feedback ID in product_insights resolves to verbatim text in df_voc."""
    df_voc, _, _, bundles, _, product_insights = load_data()
    voc_lookup = df_voc.set_index("feedback_id")["feedback_text"].to_dict()
    bundle_map = {b["cluster_id"]: b for b in bundles}

    for ins in product_insights:
        cid = ins["cluster_id"]
        b = bundle_map[cid]
        bundle_fids = {
            item["feedback_id"]
            for item in (b.get("evidence_items") or b.get("supporting_feedback", []))
        }

        cited_ids = ins["supporting_feedback_ids"]
        assert len(cited_ids) >= 3
        for fid in cited_ids:
            assert fid in bundle_fids
            assert fid in voc_lookup
            assert len(voc_lookup[fid]) > 0


def test_cluster_6_mixed_warning_condition():
    """Verify Cluster 6 is recognized as Mixed Customer Feedback."""
    _, _, _, _, _, product_insights = load_data()
    c6 = next(ins for ins in product_insights if ins["cluster_id"] == 6)
    assert "Mixed" in c6["theme_name"]
    assert c6["decision"] == "BUILD"


def test_decision_colors_and_badge_renderer():
    """Verify semantic decision color mappings and HTML badge rendering."""
    for decision in ["BUILD", "INVESTIGATE", "MONITOR", "DON'T BUILD YET"]:
        assert decision in DECISION_COLORS
        badge_html = render_decision_badge(decision)
        assert decision in badge_html
        assert DECISION_COLORS[decision] in badge_html

    # Test Mixed signal badge
    mixed_badge = render_decision_badge("BUILD", is_mixed=True)
    assert "MIXED SIGNAL" in mixed_badge


def test_data_consistency_across_artifacts():
    """Audit data consistency across all 8 clusters between voc_feedback, bundles, and priorities."""
    df_voc, df_clusters, theme_defs, bundles, priority_assessments, product_insights = load_data()
    merged = pd.merge(df_voc, df_clusters, on="feedback_id")

    assert len(merged) == 800

    for cid in range(8):
        c_df = merged[merged["cluster_id"] == cid]
        b = next(x for x in bundles if x["cluster_id"] == cid)
        p = next(x for x in priority_assessments if x["cluster_id"] == cid)
        ins = next(x for x in product_insights if x["cluster_id"] == cid)

        # Volume consistency
        assert len(c_df) == b["evidence_count"]
        assert len(c_df) == p["evidence_count"]

        # Sentiment consistency
        s_counts = c_df["sentiment"].str.lower().value_counts().to_dict()
        b_s_counts = b.get("sentiment_counts", {})
        for s_key, s_val in s_counts.items():
            assert b_s_counts.get(s_key, 0) == s_val

        # Source diversity consistency
        assert c_df["source_type"].nunique() == b["source_diversity_count"]

        # Known segments count
        known_segs = c_df[c_df["customer_segment"] != "Unknown"]["customer_segment"].nunique()
        assert known_segs == p["known_segment_count"]


def test_format_recommended_action_helper():
    """Verify presentation formatter produces lead + at least 2 key moves for all 8 themes."""
    from app.streamlit_app import format_recommended_action

    _, _, _, _, _, product_insights = load_data()

    for ins in product_insights:
        cid = ins["cluster_id"]
        lead, moves = format_recommended_action(ins["recommended_action"], cluster_id=cid)
        assert len(lead) > 0
        assert isinstance(moves, list)
        assert len(moves) >= 2, f"Cluster {cid} has fewer than 2 key moves"
        for m in moves:
            assert len(m.strip()) > 0


def test_render_recommended_action_card_no_raw_html():
    """Verify render_recommended_action_card produces valid HTML with sequential Key Moves across all 8 themes."""
    from app.streamlit_app import render_recommended_action_card

    _, _, _, _, priority_assessments, product_insights = load_data()
    p_map = {p["cluster_id"]: p["decision"] for p in priority_assessments}

    for ins in product_insights:
        cid = ins["cluster_id"]
        decision = p_map[cid]
        card_html = render_recommended_action_card(decision, ins["recommended_action"], cluster_id=cid)

        assert "action-card" in card_html
        assert "action-header" in card_html
        assert "action-body" in card_html
        assert "Key Moves" in card_html
        assert "01" in card_html
        assert "02" in card_html

        # Verify no 4+ space leading indentation on any line (which triggers Markdown code blocks)
        for line in card_html.split("\n"):
            assert not line.startswith("    "), f"Cluster {cid} produced 4-space indented line: {line}"

        # Decision alignment checks
        if decision == "DON'T BUILD YET":
            assert "Defer" in card_html or "defer" in card_html or "avoid" in card_html or "friction" in card_html
        elif decision == "INVESTIGATE":
            assert "discovery" in card_html or "discovery" in card_html.lower() or "quality" in card_html.lower()
        elif decision == "MONITOR":
            assert "telemetry" in card_html.lower() or "monitor" in card_html.lower()
