"""
Voice of Customer Copilot — Streamlit MVP Dashboard
PM-Level Intelligence: Decision Launcher, Progressive Disclosure & Distinct Information Architecture.

Consumes canonical analytical artifacts:
1. data/processed/voc_feedback.csv
2. data/processed/theme_clusters.csv
3. data/processed/theme_definitions.json
4. data/processed/evidence_bundles.json
5. data/processed/priority_assessments.json
6. data/processed/product_insights.json
"""

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Voice of Customer Copilot",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

VOC_FEEDBACK_PATH = PROCESSED_DIR / "voc_feedback.csv"
CLUSTERS_PATH = PROCESSED_DIR / "theme_clusters.csv"
THEME_DEFS_PATH = PROCESSED_DIR / "theme_definitions.json"
BUNDLES_PATH = PROCESSED_DIR / "evidence_bundles.json"
PRIORITY_PATH = PROCESSED_DIR / "priority_assessments.json"
INSIGHTS_PATH = PROCESSED_DIR / "product_insights.json"


# Semantic Color Mapping
DECISION_COLORS = {
    "BUILD": "#10B981",          # Emerald Green
    "INVESTIGATE": "#F59E0B",    # Amber
    "MONITOR": "#3B82F6",        # Blue
    "DON'T BUILD YET": "#64748B" # Slate Gray
}

DECISION_BG_COLORS = {
    "BUILD": "rgba(16, 185, 129, 0.08)",
    "INVESTIGATE": "rgba(245, 158, 11, 0.08)",
    "MONITOR": "rgba(59, 130, 246, 0.08)",
    "DON'T BUILD YET": "rgba(100, 116, 139, 0.08)",
}

DECISION_BORDER_COLORS = {
    "BUILD": "rgba(16, 185, 129, 0.35)",
    "INVESTIGATE": "rgba(245, 158, 11, 0.35)",
    "MONITOR": "rgba(59, 130, 246, 0.35)",
    "DON'T BUILD YET": "rgba(100, 116, 139, 0.35)",
}

# Premium SaaS Design System CSS
CUSTOM_CSS = """
<style>
    /* Global Typography & Spacing */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #0f172a;
    }
    
    .app-header {
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid #e2e8f0;
    }
    .app-title {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -0.7px;
        color: #0f172a;
        margin: 0;
    }
    .app-subtitle {
        font-size: 16.5px;
        font-weight: 500;
        color: #475569;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .app-provenance {
        font-size: 13.5px;
        color: #64748b;
        font-weight: 400;
    }

    /* Decision Pill Badge */
    .decision-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border: 1px solid transparent;
    }

    /* Hero Metric Block */
    .hero-metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .hero-metric-title {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }
    .hero-metric-val {
        font-size: 30px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .hero-metric-desc {
        font-size: 12.5px;
        color: #64748b;
        margin-top: 4px;
    }

    /* Build Now Agenda Item */
    .build-agenda-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 18px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .build-agenda-row:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    .build-row-index {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 17px;
        font-weight: 800;
        color: #10b981;
        margin-right: 18px;
        min-width: 28px;
    }
    .build-row-title {
        font-size: 16.5px;
        font-weight: 700;
        color: #0f172a;
        flex-grow: 1;
    }

    /* Other Signals Box */
    .other-signals-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-top: 8px;
    }

    /* Exploration Theme Card */
    .theme-grid-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .theme-grid-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
    }
    .theme-grid-name {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        margin: 12px 0 8px 0;
        line-height: 1.35;
    }
    .theme-grid-def {
        font-size: 14.5px;
        line-height: 1.55;
        color: #475569;
        margin-bottom: 16px;
        min-height: 42px;
    }
    .theme-grid-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        font-size: 13px;
        font-weight: 600;
        color: #334155;
        border-top: 1px solid #f1f5f9;
        padding-top: 12px;
    }

    /* Action Card in Theme Detail */
    .action-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .action-header {
        font-size: 12.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748b;
        margin-bottom: 10px;
    }
    .action-body {
        font-size: 15px;
        line-height: 1.6;
        color: #1e293b;
    }
    .key-move-item {
        display: flex;
        align-items: flex-start;
        margin-top: 8px;
        font-size: 14.5px;
        line-height: 1.55;
        color: #334155;
    }
    .key-move-num {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-weight: 700;
        color: #10b981;
        margin-right: 10px;
        min-width: 24px;
    }

    /* KPI Metric Card in Theme Detail */
    .kpi-metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .kpi-val {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 11.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748b;
        margin-top: 4px;
    }
    .kpi-desc {
        font-size: 12.5px;
        font-weight: 600;
        color: #334155;
        margin-top: 4px;
    }

    /* Verbatim Evidence Quote Card */
    .evidence-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .evidence-id-tag {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12px;
        font-weight: 700;
        background: #eff6ff;
        color: #1d4ed8;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid #dbeafe;
    }
    .evidence-quote {
        font-size: 15.5px;
        line-height: 1.6;
        color: #0f172a;
        margin: 12px 0;
        font-weight: 450;
    }
    .evidence-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        font-size: 12.5px;
        color: #475569;
    }
    .tag-pill {
        background: #f1f5f9;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 500;
    }

    /* Mixed Signal Prominent Warning Callout */
    .mixed-callout {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 5px solid #f59e0b;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .mixed-callout-title {
        font-size: 13.5px;
        font-weight: 800;
        color: #92400e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .mixed-callout-text {
        font-size: 14.5px;
        color: #78350f;
        line-height: 1.55;
    }

    /* Trust banner */
    .trust-banner {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Feedback Card in Card View */
    .feedback-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load and cache all authoritative datasets and artifacts."""
    required_files = [
        VOC_FEEDBACK_PATH,
        CLUSTERS_PATH,
        THEME_DEFS_PATH,
        BUNDLES_PATH,
        PRIORITY_PATH,
        INSIGHTS_PATH,
    ]
    for path in required_files:
        if not path.exists():
            st.error(f"Authoritative artifact missing: `{path.name}`.")
            st.stop()

    df_voc = pd.read_csv(VOC_FEEDBACK_PATH)
    df_clusters = pd.read_csv(CLUSTERS_PATH)

    with open(THEME_DEFS_PATH, "r", encoding="utf-8") as f:
        theme_defs = json.load(f)
    with open(BUNDLES_PATH, "r", encoding="utf-8") as f:
        bundles = json.load(f)
    with open(PRIORITY_PATH, "r", encoding="utf-8") as f:
        priority_assessments = json.load(f)
    with open(INSIGHTS_PATH, "r", encoding="utf-8") as f:
        product_insights = json.load(f)

    return df_voc, df_clusters, theme_defs, bundles, priority_assessments, product_insights




def render_decision_badge(decision: str, is_mixed: bool = False) -> str:
    """Return inline HTML for a clean decision pill badge."""
    if is_mixed:
        return '<span class="decision-pill" style="color: #b45309; background-color: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.40);">● BUILD · MIXED SIGNAL</span>'
    
    color = DECISION_COLORS.get(decision, "#64748B")
    bg = DECISION_BG_COLORS.get(decision, "rgba(100, 116, 139, 0.10)")
    border = DECISION_BORDER_COLORS.get(decision, "rgba(100, 116, 139, 0.35)")
    return f'<span class="decision-pill" style="color: {color}; background-color: {bg}; border-color: {border};">● {decision}</span>'


# Curated, normalized presentation map decomposing authoritative V4 recommendations into concise lead + 3 concrete Key Moves
NORMALIZED_ACTION_PLANS: Dict[int, Dict[str, Any]] = {
    0: {
        "lead": "Execute targeted qualitative discovery and prototype testing before committing to full feature buildout.",
        "moves": [
            "Validate high-impact assistant concepts such as spec comparisons and review summarization through user prototypes.",
            "Evaluate technical feasibility, model latency, and catalog API integration requirements.",
            "Measure user adoption intent and task-completion lift prior to allocating dedicated feature engineering."
        ]
    },
    1: {
        "lead": "Establish automated telemetry monitoring and threshold alerts to protect high baseline satisfaction as platform traffic scales.",
        "moves": [
            "Track one-click payment success rates (UPI, Apple Pay) and latency across checkout steps.",
            "Monitor navigation drop-off rates and fulfillment SLA compliance with automated threshold alerting.",
            "Maintain regression test suites for critical purchasing funnels without diverting core feature roadmap resources."
        ]
    },
    2: {
        "lead": "Build a centralized, real-time Estimated Delivery Date (EDD) service and proactive transit notification pipeline.",
        "moves": [
            "Unify estimated delivery calculations across product pages, checkout, tracking dashboards, and customer support.",
            "Implement automated proactive notifications whenever shipment statuses or transit timelines regress.",
            "Integrate real-time carrier SLA data to eliminate discrepant delivery expectations between surfaces."
        ]
    },
    3: {
        "lead": "Initiate targeted user discovery and batch quality analysis to identify specific build components driving mediocre ratings.",
        "moves": [
            "Conduct discovery interviews with customers giving average ratings to isolate specific material or finish deficiencies.",
            "Perform exploratory quality analysis across production batches and component supplier tiers.",
            "Define minimum build-quality benchmarks before deciding on product redesign or supplier intervention."
        ]
    },
    4: {
        "lead": "Execute a root-cause quality engineering audit and deploy component upgrades to address premature breakdown.",
        "moves": [
            "Audit hardware and material failure logs to pinpoint primary premature component degradation points.",
            "Establish elevated durability and stress test suites prior to manufacturing and batch release.",
            "Deploy component-level defect fixes and supplier quality standards to eliminate premature failures."
        ]
    },
    5: {
        "lead": "Defer engineering investment and preserve the existing high-performing product experience.",
        "moves": [
            "Avoid allocating engineering or design resources to this theme given absence of customer friction.",
            "Track ongoing sentiment and rating telemetry to ensure satisfaction remains consistently high.",
            "Revisit resource allocation only if new customer friction or negative sentiment emerges."
        ]
    },
    6: {
        "lead": "Decompose this heterogeneous priority area into targeted modular engineering tracks rather than a single monolithic build.",
        "moves": [
            "Search Relevance Track: Fix multi-attribute search ranking and implement dynamic client-side filter updates.",
            "Onboarding UX Track: Reduce upfront permission requests and simplify Personal vs. Business account selection.",
            "Fulfillment Links Track: Update SMS fulfillment integrations to link directly to carrier tracking status pages."
        ]
    },
    7: {
        "lead": "Build resilient checkout payment flow enhancements and automated background transaction reconciliation.",
        "moves": [
            "Implement automatic background payment reconciliation and idempotent webhook listeners for pending gateway transactions.",
            "Add app crash recovery and state restoration during biometric authentication and gateway redirects.",
            "Provide clear real-time transaction status UI messaging to eliminate customer anxiety during payment processing."
        ]
    }
}


def format_recommended_action(action_text: str, cluster_id: Optional[int] = None) -> Tuple[str, List[str]]:
    """Presentation-only formatter: returns normalized lead sentence + 2-3 concrete Key Moves."""
    if cluster_id is not None and cluster_id in NORMALIZED_ACTION_PLANS:
        plan = NORMALIZED_ACTION_PLANS[cluster_id]
        return plan["lead"], list(plan["moves"])

    # Fallback parser for dynamic action strings
    if "1)" in action_text and "2)" in action_text:
        lead_part = action_text.split("1)")[0].strip().rstrip(":")
        moves_part = action_text[action_text.index("1)"):]
        raw_moves = re.split(r'\s*\d+\)\s*', moves_part)
        moves = []
        for m in raw_moves:
            cleaned = m.strip().rstrip(";,.")
            if cleaned.endswith("; and"):
                cleaned = cleaned[:-5]
            elif cleaned.startswith("and "):
                cleaned = cleaned[4:]
            if cleaned:
                moves.append(cleaned)
        return lead_part, moves[:3]

    sentences = [s.strip() for s in re.split(r'\.\s+(?=[A-Z])', action_text) if s.strip()]
    if len(sentences) > 1:
        lead = sentences[0] if sentences[0].endswith(".") else sentences[0] + "."
        moves = [s if s.endswith(".") else s + "." for s in sentences[1:]]
        return lead, moves[:3]

    match = re.search(r'^(.*?[^,]),\s+(?:and\s+)?(implement\s+.*|establish\s+.*|deploy\s+.*)$', action_text, re.IGNORECASE)
    if match:
        lead = match.group(1).strip().rstrip(".") + "."
        second_move = match.group(2).strip().rstrip(".")
        second_move = second_move[0].upper() + second_move[1:] + "."
        return lead, [second_move]

    return action_text, []


def render_recommended_action_card(decision: str, raw_action: str, cluster_id: Optional[int] = None) -> str:
    """Return clean, unindented inline HTML for the Recommended Product Action card with robust Key Moves."""
    decision_color = DECISION_COLORS.get(decision, "#10B981")
    lead_action, key_moves = format_recommended_action(raw_action, cluster_id=cluster_id)
    escaped_lead = html.escape(lead_action)
    
    moves_html = ""
    if key_moves:
        moves_items = []
        for idx, km in enumerate(key_moves, 1):
            escaped_km = html.escape(km)
            moves_items.append(
                f'<div class="key-move-item">'
                f'<span class="key-move-num">0{idx}</span>'
                f'<span>{escaped_km}</span>'
                f'</div>'
            )
        moves_html = (
            '<div style="margin-top: 14px; font-weight: 700; font-size: 12px; color: #64748b; text-transform: uppercase;">Key Moves</div>'
            + "".join(moves_items)
        )
    
    return (
        f'<div class="action-card" style="border-left: 4px solid {decision_color};">'
        f'<div class="action-header" style="color: {decision_color};">🚀 Recommended Product Action</div>'
        f'<div class="action-body">{escaped_lead}</div>'
        f'{moves_html}'
        f'</div>'
    )




# ==============================================================================
# 1. OVERVIEW PAGE (Executive Decision Launcher)
# ==============================================================================
def render_overview_page(priority_assessments: List[Dict[str, Any]]):
    """Render the simplified Executive Landing Page answering: What should I focus on?"""
    # Header
    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-title">VOICE OF CUSTOMER COPILOT</h1>
            <div class="app-subtitle">Customer intelligence for product decisions</div>
            <div class="app-provenance">800 customer feedback records • 8 discovered themes</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Hero Decision Summary Distribution: "4 ACTIONS NEED ATTENTION"
    decisions = [p["decision"] for p in priority_assessments]
    b_cnt = decisions.count("BUILD")
    i_cnt = decisions.count("INVESTIGATE")
    m_cnt = decisions.count("MONITOR")
    d_cnt = decisions.count("DON'T BUILD YET")

    st.markdown("### 4 ACTIONS NEED ATTENTION")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="hero-metric-card" style="border-top: 3px solid {DECISION_COLORS['BUILD']};">
                <div class="hero-metric-title" style="color: {DECISION_COLORS['BUILD']};">BUILD</div>
                <div class="hero-metric-val" style="color: {DECISION_COLORS['BUILD']};">{b_cnt}</div>
                <div class="hero-metric-desc">Immediate product intervention</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="hero-metric-card" style="border-top: 3px solid {DECISION_COLORS['INVESTIGATE']};">
                <div class="hero-metric-title" style="color: {DECISION_COLORS['INVESTIGATE']};">INVESTIGATE</div>
                <div class="hero-metric-val" style="color: {DECISION_COLORS['INVESTIGATE']};">{i_cnt}</div>
                <div class="hero-metric-desc">Targeted user discovery</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="hero-metric-card" style="border-top: 3px solid {DECISION_COLORS['MONITOR']};">
                <div class="hero-metric-title" style="color: {DECISION_COLORS['MONITOR']};">MONITOR</div>
                <div class="hero-metric-val" style="color: {DECISION_COLORS['MONITOR']};">{m_cnt}</div>
                <div class="hero-metric-desc">Telemetry & signal tracking</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="hero-metric-card" style="border-top: 3px solid {DECISION_COLORS["DON'T BUILD YET"]};">
                <div class="hero-metric-title" style="color: {DECISION_COLORS["DON'T BUILD YET"]};">DON'T BUILD</div>
                <div class="hero-metric-val" style="color: {DECISION_COLORS["DON'T BUILD YET"]};">{d_cnt}</div>
                <div class="hero-metric-desc">Defer investment</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.write("")

    # 2. BUILD FOCUS SECTION (Interactive Numbered List of the 4 BUILD themes only)
    st.markdown("### 🎯 BUILD NOW")
    st.caption("Customer themes currently recommended for product intervention.")

    build_themes = [p for p in sorted(priority_assessments, key=lambda x: x["priority_score"], reverse=True) if p["decision"] == "BUILD"]

    for idx, p in enumerate(build_themes, start=1):
        is_mixed = (p["cluster_id"] == 6 or "Mixed" in p["theme_name"])
        badge_html = render_decision_badge(p["decision"], is_mixed=is_mixed)
        
        c_left, c_btn = st.columns([4, 1])
        with c_left:
            st.markdown(
                f"""
                <div class="build-agenda-row">
                    <span class="build-row-index">0{idx}</span>
                    <span class="build-row-title">{p['theme_name']}</span>
                    <span style="margin-left: 12px;">{badge_html}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_btn:
            if st.button(f"Explore →", key=f"btn_overview_build_{p['cluster_id']}", use_container_width=True):
                st.session_state["selected_theme_id"] = p["cluster_id"]
                st.session_state["nav_selection"] = "Themes"
                st.rerun()

    st.write("")
    st.markdown("---")

    # 3. OTHER SIGNALS (Compact, Interactive Secondary Action Summary)
    st.markdown("### ⚡ OTHER SIGNALS")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(
            f"""
            <div class="other-signals-box">
                <div style="font-size: 15px; font-weight: 700; color: #0f172a;">INVESTIGATE · {i_cnt}</div>
                <div style="font-size: 13px; color: #64748b; margin: 4px 0 12px 0;">Targeted user discovery required to validate friction scope.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("View Investigate Themes →", key="btn_signal_investigate", use_container_width=True):
            st.session_state["selected_theme_id"] = None
            st.session_state["theme_filter_decision"] = "INVESTIGATE"
            st.session_state["nav_selection"] = "Themes"
            st.rerun()

    with col_s2:
        st.markdown(
            f"""
            <div class="other-signals-box">
                <div style="font-size: 15px; font-weight: 700; color: #0f172a;">MONITOR · {m_cnt}</div>
                <div style="font-size: 13px; color: #64748b; margin: 4px 0 12px 0;">Track usage telemetry and user adoption trends.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("View Monitor Themes →", key="btn_signal_monitor", use_container_width=True):
            st.session_state["selected_theme_id"] = None
            st.session_state["theme_filter_decision"] = "MONITOR"
            st.session_state["nav_selection"] = "Themes"
            st.rerun()

    with col_s3:
        st.markdown(
            f"""
            <div class="other-signals-box">
                <div style="font-size: 15px; font-weight: 700; color: #0f172a;">DEFER · {d_cnt}</div>
                <div style="font-size: 13px; color: #64748b; margin: 4px 0 12px 0;">Low priority friction; defer engineering effort.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("View Deferred Themes →", key="btn_signal_defer", use_container_width=True):
            st.session_state["selected_theme_id"] = None
            st.session_state["theme_filter_decision"] = "DON'T BUILD YET"
            st.session_state["nav_selection"] = "Themes"
            st.rerun()


# ==============================================================================
# 2. THEMES PAGE (Exploration Card Grid & Deep-Dive Detail View)
# ==============================================================================
def render_themes_page(
    priority_assessments: List[Dict[str, Any]],
    product_insights: List[Dict[str, Any]],
    bundles: List[Dict[str, Any]],
    df_voc: pd.DataFrame,
    df_clusters: pd.DataFrame
):
    """Render exploration card grid of all 8 themes or a progressive deep-dive detail view."""
    insight_map = {ins["cluster_id"]: ins for ins in product_insights}
    bundle_map = {b["cluster_id"]: b for b in bundles}
    priority_map = {p["cluster_id"]: p for p in priority_assessments}

    # Header
    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-title">CUSTOMER THEMES</h1>
            <div class="app-subtitle">What customer problems did we discover across 800 feedback records?</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Check if a specific theme is currently selected for deep-dive
    selected_cid = st.session_state.get("selected_theme_id", None)

    if selected_cid is not None and selected_cid in priority_map:
        # DETAIL DEEP-DIVE VIEW ("Reasoning")
        p = priority_map[selected_cid]
        ins = insight_map[selected_cid]
        b = bundle_map[selected_cid]
        is_mixed = (selected_cid == 6 or "Mixed" in p["theme_name"])

        if st.button("← Back to all themes", key="btn_back_themes"):
            st.session_state["selected_theme_id"] = None
            st.rerun()

        st.write("")
        badge_html = render_decision_badge(p["decision"], is_mixed=is_mixed)
        
        st.markdown(
            f"""
            <div style="margin-bottom: 22px;">
                <div style="margin-bottom: 8px;">{badge_html}</div>
                <h2 style="font-size: 28px; font-weight: 800; margin: 4px 0 8px 0; color: #0f172a; letter-spacing: -0.4px;">{p['theme_name']}</h2>
                <div style="font-size: 15.5px; color: #475569;">{p.get('theme_definition', '')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Cluster 6 mixed signal prominent callout
        if is_mixed:
            st.markdown(
                f"""
                <div class="mixed-callout">
                    <div class="mixed-callout-title">⚠️ MIXED SIGNAL — DECOMPOSITION REQUIRED</div>
                    <div class="mixed-callout-text">
                        This cluster contains multiple distinct customer needs (Search Relevance, Onboarding UX, and SMS Tracking Links).
                        While the aggregate priority score is high (<b>{p['priority_score']:.3f}</b>, triggering BUILD), the evidence must be <b>decomposed into separate product tracks</b> before committing to an engineering build.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Core PM Value: 1. Customer Problem & 2. Recommended Action
        col1, col2 = st.columns(2)
        with col1:
            escaped_problem = html.escape(ins.get('customer_problem', ''))
            st.markdown(
                f'<div class="action-card">'
                f'<div class="action-header">🎯 Customer Problem</div>'
                f'<div class="action-body">{escaped_problem}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col2:
            action_card_html = render_recommended_action_card(p['decision'], ins.get('recommended_action', ''), cluster_id=selected_cid)
            st.markdown(action_card_html, unsafe_allow_html=True)



        # 3. Why This Matters KPI Metrics Bar (Authoritative numbers only)
        st.markdown("#### Why This Matters")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="kpi-metric-card">
                    <div class="kpi-val">{p['priority_score']:.3f}</div>
                    <div class="kpi-label">Priority Score</div>
                    <div class="kpi-desc" style="color: {DECISION_COLORS[p['decision']]};">{p['decision']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""
                <div class="kpi-metric-card">
                    <div class="kpi-val">{p['evidence_count']}</div>
                    <div class="kpi-label">Evidence Volume</div>
                    <div class="kpi-desc">{b['source_diversity_count']} source channels</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f"""
                <div class="kpi-metric-card">
                    <div class="kpi-val">{p['mean_severity']:.2f} / 5</div>
                    <div class="kpi-label">Mean Severity</div>
                    <div class="kpi-desc">{p['severity_4_or_5_count']} severe reports (sev ≥4)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m4:
            segment_label = f"{p['known_segment_count']} known segments represented"
            st.markdown(
                f"""
                <div class="kpi-metric-card">
                    <div class="kpi-val">{p['known_segment_count']}</div>
                    <div class="kpi-label">User Segments</div>
                    <div class="kpi-desc">{segment_label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # 4. Tab-Based Progressive Disclosure: [ Evidence | Analysis | Limitations ]
        tab_evidence, tab_analysis, tab_limitations = st.tabs(["🔎 Evidence", "📊 Analysis", "⚠️ Limitations"])

        # TAB 1: EVIDENCE
        with tab_evidence:
            st.caption(f"Verbatim customer feedback records grounding this insight (Total cited: {len(ins.get('supporting_feedback_ids', []))}).")
            
            cited_ids = ins.get("supporting_feedback_ids", [])
            voc_lookup = df_voc.set_index("feedback_id").to_dict("index")
            bundle_evidence = {item["feedback_id"]: item for item in (b.get("evidence_items") or b.get("supporting_feedback", []))}

            # Show top 3 records by default
            for idx, fid in enumerate(cited_ids[:3], start=1):
                raw = voc_lookup.get(fid, {})
                emb_rec = bundle_evidence.get(fid, {})

                text = raw.get("feedback_text", emb_rec.get("feedback_text", ""))
                source = raw.get("source_type", "Unknown")
                sentiment = str(raw.get("sentiment", "Neutral")).capitalize()
                severity = raw.get("severity", "-")
                segment = raw.get("customer_segment", "Unknown")

                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="evidence-id-tag">CUSTOMER EVIDENCE #{idx} • {fid}</span>
                            <span style="font-size: 11.5px; color: #64748b; font-weight: 600;">Original customer feedback</span>
                        </div>
                        <div class="evidence-quote">"{text}"</div>
                        <div class="evidence-tags">
                            <span class="tag-pill">Source: <b>{source}</b></span>
                            <span class="tag-pill">Sentiment: <b>{sentiment}</b></span>
                            <span class="tag-pill">Severity: <b>{severity}/5</b></span>
                            <span class="tag-pill">Segment: <b>{segment}</b></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if len(cited_ids) > 3:
                with st.expander(f"Show {len(cited_ids) - 3} more supporting records"):
                    for idx, fid in enumerate(cited_ids[3:], start=4):
                        raw = voc_lookup.get(fid, {})
                        emb_rec = bundle_evidence.get(fid, {})

                        text = raw.get("feedback_text", emb_rec.get("feedback_text", ""))
                        source = raw.get("source_type", "Unknown")
                        sentiment = str(raw.get("sentiment", "Neutral")).capitalize()
                        severity = raw.get("severity", "-")
                        segment = raw.get("customer_segment", "Unknown")

                        st.markdown(
                            f"""
                            <div class="evidence-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span class="evidence-id-tag">CUSTOMER EVIDENCE #{idx} • {fid}</span>
                                    <span style="font-size: 11.5px; color: #64748b; font-weight: 600;">Original customer feedback</span>
                                </div>
                                <div class="evidence-quote">"{text}"</div>
                                <div class="evidence-tags">
                                    <span class="tag-pill">Source: <b>{source}</b></span>
                                    <span class="tag-pill">Sentiment: <b>{sentiment}</b></span>
                                    <span class="tag-pill">Severity: <b>{severity}/5</b></span>
                                    <span class="tag-pill">Segment: <b>{segment}</b></span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        # TAB 2: ANALYSIS
        with tab_analysis:
            theme_fids = set(df_clusters[df_clusters["cluster_id"] == selected_cid]["feedback_id"])
            theme_voc_df = df_voc[df_voc["feedback_id"].isin(theme_fids)]

            ch1, ch2, ch3 = st.columns(3)
            with ch1:
                # 1. Customer Signal
                sent_counts = theme_voc_df["sentiment"].str.capitalize().value_counts().reset_index()
                sent_counts.columns = ["Sentiment", "Count"]
                color_map = {"Positive": "#10B981", "Neutral": "#64748B", "Negative": "#EF4444"}
                fig_sent = px.pie(
                    sent_counts,
                    names="Sentiment",
                    values="Count",
                    title="CUSTOMER SIGNAL: How are customers feeling?",
                    color="Sentiment",
                    color_discrete_map=color_map,
                    hole=0.45
                )
                fig_sent.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=210, showlegend=True)
                st.plotly_chart(fig_sent, use_container_width=True)

            with ch2:
                # 2. Source Signal
                src_counts = theme_voc_df["source_type"].value_counts().reset_index()
                src_counts.columns = ["Source", "Count"]
                fig_src = px.bar(
                    src_counts,
                    x="Count",
                    y="Source",
                    orientation="h",
                    title="SOURCE SIGNAL: Where is it reported?",
                    color_discrete_sequence=["#3B82F6"]
                )
                fig_src.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=210, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_src, use_container_width=True)

            with ch3:
                # 3. Severity Signal
                sev_counts = theme_voc_df["severity"].dropna().astype(int).value_counts().sort_index().reset_index()
                sev_counts.columns = ["Severity", "Count"]
                fig_sev = px.bar(
                    sev_counts,
                    x="Severity",
                    y="Count",
                    title="SEVERITY SIGNAL: How painful is it?",
                    color_discrete_sequence=["#F59E0B"]
                )
                fig_sev.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=210)
                st.plotly_chart(fig_sev, use_container_width=True)

            st.write("")
            st.markdown("##### Deterministic Prioritization Calculation")
            st.caption("Priority = Frequency (40%) × Severity (40%) × Segment Impact (20%) × Evidence Strength × Evidence Diversity")
            
            p1, p2 = st.columns([3, 1])
            with p1:
                st.write(f"**Frequency Score**: `{p['frequency_score']:.2f}`")
                st.progress(min(1.0, float(p['frequency_score'])))
                st.write(f"**Severity Score**: `{p['severity_score']:.2f}`")
                st.progress(min(1.0, float(p['severity_score'])))
                st.write(f"**Segment Impact**: `{p['segment_impact_score']:.2f}`")
                st.progress(min(1.0, float(p['segment_impact_score'])))
                st.write(f"**Evidence Strength**: `{p['evidence_strength']:.2f}`")
                st.progress(min(1.0, float(p['evidence_strength'])))
                st.write(f"**Diversity Score**: `{p['evidence_diversity_score']:.2f}`")
                st.progress(min(1.0, float(p['evidence_diversity_score'])))
            with p2:
                st.markdown(
                    f"""
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; text-align: center;">
                        <div style="font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase;">Final Priority</div>
                        <div style="font-size: 34px; font-weight: 800; color: #0f172a; margin: 8px 0;">{p['priority_score']:.3f}</div>
                        <div>{badge_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # TAB 3: LIMITATIONS
        with tab_limitations:
            st.markdown("##### Analytical Context & Caveats")
            st.markdown(
                f"""
                • **Segment Metadata Limitation**: {p.get('segment_limitation', 'Standard distribution across segments.')}<br>
                • **AI Interpretation Confidence**: {ins.get('confidence', 0.90):.0%}<br>
                • **Contradiction Analysis**: {b.get('contradiction_summary', 'No contradiction detected.')}<br>
                • **Analytical Notes**: {ins.get('limitations', 'No additional caveats.')}
                """,
                unsafe_allow_html=True
            )

    else:
        # GRID VIEW OF ALL 8 THEMES ("Exploration")
        # Search and Decision Filter Pills
        col_s, col_p = st.columns([1, 2])
        with col_s:
            search_theme = st.text_input("🔍 Search themes...", placeholder="Filter by theme name or keyword...")
        with col_p:
            decision_options = ["ALL", "BUILD", "INVESTIGATE", "MONITOR", "DON'T BUILD YET"]
            current_filter = st.session_state.get("theme_filter_decision", "ALL")
            selected_decision = st.radio(
                "Filter by Decision",
                decision_options,
                index=decision_options.index(current_filter) if current_filter in decision_options else 0,
                horizontal=True,
                label_visibility="collapsed"
            )
            st.session_state["theme_filter_decision"] = selected_decision

        sorted_themes = sorted(priority_assessments, key=lambda x: x["priority_score"], reverse=True)
        if selected_decision != "ALL":
            sorted_themes = [p for p in sorted_themes if p["decision"] == selected_decision]
        if search_theme.strip():
            sorted_themes = [p for p in sorted_themes if search_theme.lower() in p["theme_name"].lower() or search_theme.lower() in p.get("theme_definition", "").lower()]

        st.write("")
        st.caption(f"Displaying {len(sorted_themes)} of 8 discovered customer themes")
        
        # Responsive 2-column card grid
        col_left, col_right = st.columns(2)
        for idx, p in enumerate(sorted_themes):
            target_col = col_left if idx % 2 == 0 else col_right
            ins = insight_map.get(p["cluster_id"], {})
            is_mixed = (p["cluster_id"] == 6 or "Mixed" in p["theme_name"])
            badge_html = render_decision_badge(p["decision"], is_mixed=is_mixed)

            with target_col:
                st.markdown(
                    f"""
                    <div class="theme-grid-card" style="border-left: 4px solid {DECISION_COLORS.get(p['decision'], '#64748b')};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            {badge_html}
                            <span style="font-size: 13.5px; font-weight: 700; color: #0f172a;">Priority: {p['priority_score']:.3f}</span>
                        </div>
                        <div class="theme-grid-name">{p['theme_name']}</div>
                        <div class="theme-grid-def">{ins.get('customer_problem', p.get('theme_definition', ''))}</div>
                        <div class="theme-grid-meta">
                            <span>Volume: <b>{p['evidence_count']} records</b></span>
                            <span>Severity: <b>{p['mean_severity']:.2f} / 5</b></span>
                            <span>Segments: <b>{p['known_segment_count']} tiers</b></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button("Explore →", key=f"btn_theme_explore_{p['cluster_id']}", use_container_width=True):
                    st.session_state["selected_theme_id"] = p["cluster_id"]
                    st.rerun()


# ==============================================================================
# 3. EVIDENCE PAGE (Dedicated Evidence Verification Experience)
# ==============================================================================
def render_evidence_page(
    priority_assessments: List[Dict[str, Any]],
    bundles: List[Dict[str, Any]],
    product_insights: List[Dict[str, Any]],
    df_voc: pd.DataFrame
):
    """Render dedicated Evidence inspection page with progressive disclosure."""
    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-title">TRACEABLE CUSTOMER EVIDENCE</h1>
            <div class="app-subtitle">Verify the exact customer feedback behind every product insight</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    theme_options = {f"{p['theme_name']} ({p['decision']})": p["cluster_id"] for p in sorted(priority_assessments, key=lambda x: x["priority_score"], reverse=True)}
    selected_theme_name = st.selectbox("🎯 Select Theme to Verify Supporting Evidence", list(theme_options.keys()), index=0)
    selected_cid = theme_options[selected_theme_name]

    p = next(item for item in priority_assessments if item["cluster_id"] == selected_cid)
    b = next(item for item in bundles if item["cluster_id"] == selected_cid)
    ins = next(item for item in product_insights if item["cluster_id"] == selected_cid)
    is_mixed = (selected_cid == 6 or "Mixed" in p["theme_name"])

    # Core Trust Banner: Insight -> Exact Customer Evidence
    st.markdown(
        f"""
        <div class="trust-banner">
            <div>
                <span style="font-size: 16.5px; font-weight: 700; color: #0f172a;">{p['theme_name']}</span>
                <span style="margin-left: 10px;">{render_decision_badge(p['decision'], is_mixed=is_mixed)}</span>
            </div>
            <div style="display: flex; gap: 20px; font-size: 13.5px; font-weight: 600; color: #475569;">
                <span>Volume: <b style="color: #0f172a;">{b['evidence_count']} records</b></span>
                <span>Evidence Strength: <b style="color: #0f172a;">{b['evidence_strength']:.2f}</b></span>
                <span>Source Channels: <b style="color: #0f172a;">{b['source_diversity_count']}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Contradiction Callout
    if b.get("contradiction_flag"):
        st.info(f"⚖️ **Contradiction Signals**: {b.get('contradiction_summary', 'Mixed positive and negative feedback present.')}")
    else:
        st.success(f"✅ **Sentiment Consistency**: {b.get('contradiction_summary', 'No contradictory signals detected.')}")

    st.write("")
    
    # Evidence filter toggle: [ Cited Evidence | All Theme Feedback ]
    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        evidence_mode = st.radio("Evidence Scope", ["Cited Evidence", "All Theme Feedback"], horizontal=True, label_visibility="collapsed")

    voc_lookup = df_voc.set_index("feedback_id").to_dict("index")
    evidence_items = b.get("evidence_items") or b.get("supporting_feedback", [])

    if evidence_mode == "Cited Evidence":
        cited_ids = ins.get("supporting_feedback_ids", [])
        display_items = [item for item in evidence_items if item["feedback_id"] in cited_ids]
        if not display_items:
            display_items = evidence_items[:len(cited_ids)]
    else:
        display_items = evidence_items

    st.markdown("#### Customer Voice")
    st.caption("Verbatim customer records grounding the recommendation.")

    # Display Top 3 records by default
    for idx, item in enumerate(display_items[:3], start=1):
        fid = item["feedback_id"]
        raw = voc_lookup.get(fid, {})

        text = raw.get("feedback_text", item.get("feedback_text", ""))
        source = raw.get("source_type", "Unknown")
        sentiment = str(raw.get("sentiment", "Neutral")).capitalize()
        severity = raw.get("severity", "-")
        segment = raw.get("customer_segment", "Unknown")

        st.markdown(
            f"""
            <div class="evidence-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="evidence-id-tag">CUSTOMER EVIDENCE #{idx} • {fid}</span>
                    <span style="font-size: 11.5px; color: #64748b; font-weight: 600;">Original customer feedback</span>
                </div>
                <div class="evidence-quote">"{text}"</div>
                <div class="evidence-tags">
                    <span class="tag-pill">Source: <b>{source}</b></span>
                    <span class="tag-pill">Sentiment: <b>{sentiment}</b></span>
                    <span class="tag-pill">Severity: <b>{severity}/5</b></span>
                    <span class="tag-pill">Segment: <b>{segment}</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Progressive disclosure for remaining evidence records
    if len(display_items) > 3:
        with st.expander(f"Show {len(display_items) - 3} More Evidence Records (Total {len(display_items)})"):
            for idx, item in enumerate(display_items[3:], start=4):
                fid = item["feedback_id"]
                raw = voc_lookup.get(fid, {})

                text = raw.get("feedback_text", item.get("feedback_text", ""))
                source = raw.get("source_type", "Unknown")
                sentiment = str(raw.get("sentiment", "Neutral")).capitalize()
                severity = raw.get("severity", "-")
                segment = raw.get("customer_segment", "Unknown")

                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="evidence-id-tag">CUSTOMER EVIDENCE #{idx} • {fid}</span>
                            <span style="font-size: 11.5px; color: #64748b; font-weight: 600;">Original customer feedback</span>
                        </div>
                        <div class="evidence-quote">"{text}"</div>
                        <div class="evidence-tags">
                            <span class="tag-pill">Source: <b>{source}</b></span>
                            <span class="tag-pill">Sentiment: <b>{sentiment}</b></span>
                            <span class="tag-pill">Severity: <b>{severity}/5</b></span>
                            <span class="tag-pill">Segment: <b>{segment}</b></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ==============================================================================
# 4. FEEDBACK PAGE (Customer Feedback Explorer with Card / Table Views)
# ==============================================================================
def render_feedback_page(df_voc: pd.DataFrame, df_clusters: pd.DataFrame, theme_defs: List[Dict[str, Any]]):
    """Render the searchable customer feedback explorer with Card View and Table View."""
    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-title">CUSTOMER FEEDBACK</h1>
            <div class="app-subtitle">Explore what customers are saying across 800 records</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Merge clusters and theme names
    theme_map = {t["cluster_id"]: t["theme_name"] for t in theme_defs}
    merged_voc = pd.merge(df_voc, df_clusters[["feedback_id", "cluster_id"]], on="feedback_id", how="left")
    merged_voc["discovered_theme"] = merged_voc["cluster_id"].map(theme_map)

    # Filter Bar
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        search_query = st.text_input("🔍 Search feedback text", placeholder="Type keywords (e.g. delivery, payment, battery)...")
    with c2:
        source_filter = st.selectbox("Source Type", ["All"] + sorted(merged_voc["source_type"].dropna().unique().tolist()))
    with c3:
        theme_filter = st.selectbox("Discovered Theme", ["All"] + sorted(list(theme_map.values())))
    with c4:
        sentiment_filter = st.selectbox("Sentiment", ["All", "positive", "neutral", "negative"], format_func=lambda x: x.capitalize())

    filtered_df = merged_voc.copy()
    if search_query.strip():
        filtered_df = filtered_df[filtered_df["feedback_text"].str.contains(search_query, case=False, na=False)]
    if source_filter != "All":
        filtered_df = filtered_df[filtered_df["source_type"] == source_filter]
    if theme_filter != "All":
        filtered_df = filtered_df[filtered_df["discovered_theme"] == theme_filter]
    if sentiment_filter != "All":
        filtered_df = filtered_df[filtered_df["sentiment"].str.lower() == sentiment_filter.lower()]

    total_matches = len(filtered_df)
    page_size = 10
    total_pages = max(1, (total_matches + page_size - 1) // page_size)

    # View Mode Toggle: [ Customer Voice (Cards) | Table View ]
    c_meta, c_view, c_page = st.columns([3, 1, 1])
    with c_meta:
        st.caption(f"Showing {total_matches} of 800 records (Page {st.session_state.get('feedback_page', 1)} of {total_pages})")
    with c_view:
        view_mode = st.radio("View Mode", ["Customer Voice", "Table View"], horizontal=True, label_visibility="collapsed")
    with c_page:
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="feedback_page_input")

    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size
    page_records = filtered_df.iloc[start_idx:end_idx]

    if view_mode == "Customer Voice":
        for _, row in page_records.iterrows():
            sentiment_cap = str(row['sentiment']).capitalize()
            st.markdown(
                f"""
                <div class="feedback-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span class="evidence-id-tag">{row['feedback_id']}</span>
                        <span style="font-size: 12.5px; font-weight: 600; color: #64748b;">Theme: {row['discovered_theme']}</span>
                    </div>
                    <div style="font-size: 15.5px; line-height: 1.55; color: #0f172a; margin: 10px 0;">"{row['feedback_text']}"</div>
                    <div class="evidence-tags">
                        <span class="tag-pill">Source: <b>{row['source_type']}</b></span>
                        <span class="tag-pill">Sentiment: <b>{sentiment_cap}</b></span>
                        <span class="tag-pill">Severity: <b>{row['severity']}/5</b></span>
                        <span class="tag-pill">Segment: <b>{row['customer_segment']}</b></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.dataframe(
            page_records[[
                "feedback_id", "feedback_text", "source_type", "product_area",
                "sentiment", "severity", "customer_segment", "discovered_theme"
            ]],
            use_container_width=True,
            hide_index=True,
        )


# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================
def main():
    df_voc, df_clusters, theme_defs, bundles, priority_assessments, product_insights = load_data()

    # Sidebar Navigation - ONLY Main Product Flow
    st.sidebar.markdown("### 🧭 VOC Copilot")
    
    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = "Overview"

    nav_options = ["Overview", "Themes", "Evidence", "Feedback"]
    nav_icons = {"Overview": "🏠", "Themes": "💡", "Evidence": "🔎", "Feedback": "📂"}

    selected_nav = st.sidebar.radio(
        "Navigation",
        nav_options,
        index=nav_options.index(st.session_state["nav_selection"]) if st.session_state["nav_selection"] in nav_options else 0,
        format_func=lambda x: f"{nav_icons.get(x, '')} {x}",
        label_visibility="collapsed"
    )
    st.session_state["nav_selection"] = selected_nav

    st.sidebar.markdown("---")
    
    # Secondary Methodology & Provenance
    with st.sidebar.expander("📖 About & Methodology"):
        st.markdown(
            """
            **Voice of Customer Copilot**
            - **Volume**: 800 feedback records (350 public reviews + 450 synthetic demonstration data)
            - **Embeddings**: Gemini 1536-dim vectors
            - **Clustering**: Adaptive Semantic Centroid Merging (8 clusters, 87.5% Theme Recall)
            - **Prioritization**: Deterministic 5-factor PRD formula
            - **Insights**: Gemini with citation verification
            """
        )

    st.sidebar.caption("VOC Copilot • Product Intelligence MVP")


    # Route to selected page
    if selected_nav == "Overview":
        render_overview_page(priority_assessments)
    elif selected_nav == "Themes":
        render_themes_page(priority_assessments, product_insights, bundles, df_voc, df_clusters)
    elif selected_nav == "Evidence":
        render_evidence_page(priority_assessments, bundles, product_insights, df_voc)
    elif selected_nav == "Feedback":
        render_feedback_page(df_voc, df_clusters, theme_defs)


if __name__ == "__main__":
    main()
