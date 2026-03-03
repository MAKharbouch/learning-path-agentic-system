"""Streamlit application for generating personalized learning paths.

Modern SaaS/Dashboard UI with dark sidebar, hero banner, metric cards,
and an enhanced phase timeline. All pipeline logic is unchanged.
"""

import sys
import threading
import time
from pathlib import Path

import streamlit as st

# Add src directory to Python path so internal modules can import each other
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator.graph import run_pipeline

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Learning Path Generator",
    page_icon="🎯",
    layout="wide",
)

# -------------------------------------------------------------------------
# Database initialization (idempotent)
# -------------------------------------------------------------------------
from db.schema import init_db
init_db()

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #F1F5F9; }

.block-container {
    max-width: 900px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* [data-testid="stHeader"] { display: none; } */

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #1E1F2B !important;
}
section[data-testid="stSidebar"] * { color: #CBD5E1; }
section[data-testid="stSidebar"] hr {
    border-color: #2E3044 !important;
    margin: 1rem 0;
}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton > button {
    background: #6366F1;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.4rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    transition: background 0.2s, box-shadow 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: #4F46E5;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35);
}

[data-testid="stFormSubmitButton"] > button {
    background: #6366F1;
    color: #fff !important;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1.5rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    width: 100%;
    letter-spacing: 0.01em;
    transition: background 0.2s, box-shadow 0.2s;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: #4F46E5;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35);
}
[data-testid="stFormSubmitButton"] > button:disabled {
    background: #C7D2FE !important;
    color: #818CF8 !important;
    cursor: not-allowed;
}

/* ── Form inputs ──────────────────────────────────────────────────────── */
.stTextInput input, .stNumberInput input {
    border-radius: 8px;
    border: 1.5px solid #E2E8F0;
    background: #fff;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    padding: 0.55rem 0.8rem;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #6366F1;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12);
    outline: none;
}
.stTextArea textarea {
    border-radius: 8px;
    border: 1.5px solid #E2E8F0;
    background: #fff;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    min-height: 110px;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: #6366F1;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12);
    outline: none;
}

.stTextInput label, .stNumberInput label, .stTextArea label {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: #475569;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Metric cards ─────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    border-left: 4px solid #6366F1;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94A3B8;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 800;
    color: #1E293B;
    line-height: 1.1;
}

/* ── Form card ────────────────────────────────────────────────────────── */
[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    padding: 0rem 2rem 1.2rem !important; /* Removed top padding to align header */
    overflow: hidden; /* Ensure header radius matches form */
}

/* Hide the "Press Enter to submit form" hint on every input */
[data-testid="InputInstructions"] { display: none !important; }

/* ── Alerts ───────────────────────────────────────────────────────────── */
.stAlert {
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
}

/* ── Animations ───────────────────────────────────────────────────────── */
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in-up { animation: fadeInUp 0.4s ease both; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data readiness check
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30)
def _check_data_ready():
    """Check if DB and ChromaDB are populated enough to run the pipeline."""
    from db.connection import get_connection
    from vectorstore.chroma import get_collection_no_embeddings

    conn = get_connection()
    try:
        courses = conn.execute("SELECT COUNT(*) FROM courses").fetchone()[0]
        skills = conn.execute("SELECT COUNT(*) FROM course_skills").fetchone()[0]
    except Exception:
        courses, skills = 0, 0
    finally:
        conn.close()

    try:
        collection = get_collection_no_embeddings()
        embeddings = collection.count()
    except Exception:
        embeddings = 0

    return {"courses": courses, "skills": skills, "embeddings": embeddings}


data_status = _check_data_ready()
_data_ready = (
    data_status["courses"] > 0
    and data_status["skills"] > 0
    and data_status["embeddings"] > 0
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.5rem 0 1rem;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <span style="font-size:1.6rem;">🎯</span>
                <span style="font-size:1.1rem;font-weight:800;color:#F1F5F9;
                             letter-spacing:-0.01em;">Learning Path</span>
            </div>
            <span style="font-size:0.72rem;background:#2E3044;color:#94A3B8;
                         padding:2px 8px;border-radius:12px;font-weight:500;">
                AI-Powered Generator
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;letter-spacing:0.07em;"
        "text-transform:uppercase;color:#64748B;margin-bottom:10px;'>Data Status</p>",
        unsafe_allow_html=True,
    )

    checks = [
        ("Course catalog", data_status["courses"] > 0, f"{data_status['courses']:,} courses"),
        ("Skill extraction", data_status["skills"] > 0, f"{data_status['skills']:,} skills"),
        ("Embeddings", data_status["embeddings"] > 0, f"{data_status['embeddings']:,} vectors"),
    ]
    for label, ok, detail in checks:
        icon = "✅" if ok else "❌"
        row_bg = "#1A2E24" if ok else "#2D1A1A"
        label_color = "#86EFAC" if ok else "#FCA5A5"
        detail_color = "#64748B" if ok else "#EF4444"
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:7px 10px;border-radius:8px;margin-bottom:5px;
                        background:{row_bg};">
                <span style="font-size:0.85rem;color:{label_color};">{icon} {label}</span>
                <span style="font-size:0.72rem;color:{detail_color};">{detail}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if _data_ready:
        st.markdown(
            "<div style='margin-top:10px;padding:9px 12px;border-radius:8px;"
            "background:#1A2E24;border:1px solid #166534;'>"
            "<span style='font-size:0.82rem;color:#86EFAC;font-weight:600;'>"
            "✅ All systems ready</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='margin-top:10px;padding:9px 12px;border-radius:8px;"
            "background:#2D2208;border:1px solid #92400E;'>"
            "<span style='font-size:0.82rem;color:#FCD34D;font-weight:600;'>"
            "⚠️ Setup required</span></div>",
            unsafe_allow_html=True,
        )
        missing_steps = []
        if data_status["courses"] == 0:
            missing_steps.append("Run `ingest_data.py`")
        if data_status["skills"] == 0:
            missing_steps.append("Run `extract_skills.py`")
        if data_status["embeddings"] == 0:
            missing_steps.append("Run `embed_courses.py`")
        for step in missing_steps:
            st.markdown(
                f"<p style='font-size:0.78rem;color:#94A3B8;margin:4px 0 0 12px;'>"
                f"→ {step}</p>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;letter-spacing:0.07em;"
        "text-transform:uppercase;color:#64748B;margin-bottom:10px;'>Pipeline</p>",
        unsafe_allow_html=True,
    )
    pipeline_steps = [
        ("analyze",  "Extract skill gaps"),
        ("retrieve", "Find candidate courses"),
        ("generate", "Build learning plan"),
        ("validate", "Check constraints"),
        ("repair",   "Fix violations (×3)"),
        ("finalize", "Persist to database"),
    ]
    for step, desc in pipeline_steps:
        st.markdown(
            f"<div style='display:flex;gap:8px;align-items:flex-start;margin-bottom:6px;'>"
            f"<span style='font-size:0.72rem;background:#2E3044;color:#818CF8;"
            f"padding:1px 7px;border-radius:10px;font-weight:600;flex-shrink:0;"
            f"margin-top:1px;'>{step}</span>"
            f"<span style='font-size:0.78rem;color:#94A3B8;'>{desc}</span></div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------

st.html(
    """
    <div style="
        background: linear-gradient(135deg, #1E1F2B 0%, #312E81 60%, #1E1F2B 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.12);
    ">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:0.6rem;">
            <span style="font-size:2rem;">🎯</span>
            <h1 style="margin:0;font-size:1.9rem;font-weight:800;color:#F8FAFC;
                        letter-spacing:-0.02em;font-family:'Inter',sans-serif;">
                Learning Path Generator
            </h1>
        </div>
        <p style="margin:0;font-size:0.95rem;color:#94A3B8;max-width:600px;
                   font-family:'Inter',sans-serif;line-height:1.6;">
            Enter an employee's portal ID and learning goal. The AI pipeline analyzes
            skill gaps, retrieves matching courses, and validates a personalized roadmap.
        </p>
    </div>
    """
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

with st.form("plan_form"):
    st.html(
        """
        <div style="background:#F8FAFC; border-bottom:1px solid #E2E8F0;
                    margin: 0 -2rem 1.4rem -2rem; padding: 1rem 2rem;">
            <p style="margin:0;font-size:0.72rem;font-weight:700;letter-spacing:0.07em;
                       text-transform:uppercase;color:#94A3B8;font-family:'Inter',sans-serif;">
                New Request
            </p>
        </div>
        """
    )
    col_id, col_goal = st.columns([1, 2])
    with col_id:
        portal_id_input = st.number_input(
            "Employee Portal ID",
            min_value=1,
            step=1,
            format="%d",
            help="Enter the portal ID of the employee",
        )
    with col_goal:
        goal_text = st.text_area(
            "Learning Goal",
            placeholder="e.g., I want to learn advanced cloud architecture and DevOps practices",
        )
    submitted = st.form_submit_button(
        "⚡  Generate Learning Plan",
        disabled=not _data_ready,
    )

# ---------------------------------------------------------------------------
# Pipeline execution  (logic unchanged)
# ---------------------------------------------------------------------------

if submitted:
    if not goal_text or not goal_text.strip():
        st.error("Please enter a learning goal before generating a plan.")
    else:
        portal_id = int(portal_id_input)
        result_holder: dict[str, object] = {}

        def _run(pid, goal):
            try:
                result_holder["data"] = run_pipeline(pid, goal)
            except Exception as exc:
                result_holder["error"] = str(exc)

        thread = threading.Thread(
            target=_run, args=(portal_id, goal_text.strip()), daemon=True
        )
        thread.start()
        st.session_state["_bg_thread"] = thread
        st.session_state["_bg_result"] = result_holder
        st.session_state["_bg_portal_id"] = portal_id
        st.session_state["_bg_running"] = True
        st.rerun()

# Show progress + cancel while pipeline runs
if st.session_state.get("_bg_running"):
    thread = st.session_state["_bg_thread"]
    result_holder = st.session_state["_bg_result"]

    if thread.is_alive():
        st.html(
            """
            <div style="
                background:#fff;border:1px solid #E2E8F0;border-radius:14px;
                box-shadow:0 2px 10px rgba(0,0,0,0.06);
                padding:2.5rem;margin-top:1.5rem;text-align:center;
            ">
                <div style="display:inline-block;width:48px;height:48px;
                             border:4px solid #E2E8F0;border-top:4px solid #6366F1;
                             border-radius:50%;animation:spin 0.85s linear infinite;
                             margin-bottom:1.2rem;"></div>
                <p style="margin:0 0 6px;font-size:1.05rem;font-weight:700;color:#1E293B;
                            font-family:'Inter',sans-serif;">
                    Generating your learning plan...
                </p>
                <p style="margin:0;font-size:0.85rem;color:#94A3B8;font-family:'Inter',sans-serif;">
                    Analyzing skill gaps &rarr; Retrieving courses &rarr; Validating plan
                </p>
            </div>
            <style>@keyframes spin{to{transform:rotate(360deg)}}</style>
            """
        )
        if st.button("✕  Cancel Generation"):
            st.session_state["_bg_running"] = False
            st.toast("Generation cancelled.")
            st.rerun()
        time.sleep(1)
        st.rerun()
    else:
        portal_id = st.session_state["_bg_portal_id"]
        st.session_state["_bg_running"] = False

        if "error" in result_holder:
            st.session_state.result = {
                "error": result_holder["error"],
                "error_node": "pipeline",
            }
        else:
            st.session_state.result = result_holder.get("data")

        st.session_state.portal_id = portal_id
        st.rerun()


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

if st.session_state.get("result"):
    result = st.session_state.result

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Error case
    if result.get("error"):
        error_node = result.get("error_node", "unknown")
        error_msg = result["error"]
        st.html(
            f"""
            <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:12px;
                         padding:1.2rem 1.5rem;margin-bottom:1rem;">
                <p style="margin:0 0 4px;font-size:0.78rem;font-weight:700;color:#991B1B;
                            text-transform:uppercase;letter-spacing:0.05em;">Pipeline Error</p>
                <p style="margin:0;font-size:0.9rem;color:#7F1D1D;font-family:'Inter',sans-serif;">
                    <strong>Node:</strong> {error_node}<br>{error_msg}
                </p>
            </div>
            """
        )
    else:
        plan_response = result.get("plan_response")
        validation_result = result.get("validation_result")

        if plan_response is None:
            st.html(
                """
                <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:12px;
                             padding:1.2rem 1.5rem;">
                    <p style="margin:0;font-size:0.92rem;color:#92400E;
                                font-family:'Inter',sans-serif;">
                        ⚠️ Pipeline completed but no plan was generated — no skill gaps found.
                    </p>
                </div>
                """
            )
        else:
            # ── Metric strip ──────────────────────────────────────────────
            is_valid = validation_result.is_valid if validation_result else False

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Courses Selected", len(plan_response.courses))
            with col2:
                st.metric("Estimated Hours", f"{plan_response.total_estimated_hours:.1f} h")
            with col3:
                st.metric("Plan Status", "✅ Valid" if is_valid else "⚠️ Review")

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

            # ── Validation block ──────────────────────────────────────────
            if validation_result is not None:
                if not validation_result.is_valid:
                    n = len(validation_result.violations)
                    items_html = "".join(
                        f"<div style='display:flex;gap:8px;align-items:flex-start;"
                        f"margin-bottom:6px;'>"
                        f"<span style='font-size:0.72rem;font-weight:700;padding:2px 8px;"
                        f"border-radius:10px;flex-shrink:0;margin-top:1px;"
                        f"background:{'#FEE2E2' if v.severity == 'error' else '#FEF3C7'};"
                        f"color:{'#991B1B' if v.severity == 'error' else '#92400E'};'>"
                        f"{v.severity.upper()}</span>"
                        f"<span style='font-size:0.87rem;color:#475569;'>"
                        f"[{v.check_name}] {v.message}</span></div>"
                        for v in validation_result.violations
                    )
                    st.html(
                        f"""
                        <div style="background:#FFFBEB;border:1px solid #FDE68A;
                                     border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
                            <p style="margin:0 0 10px;font-size:0.78rem;font-weight:700;
                                        color:#92400E;text-transform:uppercase;letter-spacing:0.05em;">
                                ⚠️ {n} Validation Issue{'s' if n != 1 else ''}
                            </p>
                            {items_html}
                        </div>
                        """
                    )
                else:
                    st.html(
                        """
                        <div style="background:#F0FDF4;border:1px solid #BBF7D0;
                                     border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;">
                            <span style="font-size:0.92rem;color:#166534;font-weight:600;
                                          font-family:'Inter',sans-serif;">
                                ✅ Plan passed all validation checks
                            </span>
                        </div>
                        """
                    )

            # ── Skill coverage callout ────────────────────────────────────
            st.html(
                f"""
                <div style="background:#EEF2FF;border-left:4px solid #6366F1;
                             border-radius:0 12px 12px 0;padding:1rem 1.4rem;
                             margin-bottom:1.4rem;">
                    <p style="margin:0 0 4px;font-size:0.72rem;font-weight:700;
                                text-transform:uppercase;letter-spacing:0.06em;color:#6366F1;">
                        Skill Coverage Summary
                    </p>
                    <p style="margin:0;font-size:0.92rem;color:#3730A3;line-height:1.6;
                                font-family:'Inter',sans-serif;">
                        {plan_response.skill_coverage_summary}
                    </p>
                </div>
                """
            )

            # ── Enhanced timeline ─────────────────────────────────────────
            st.markdown(
                "<p style='font-size:0.72rem;font-weight:700;letter-spacing:0.07em;"
                "text-transform:uppercase;color:#94A3B8;margin-bottom:0.8rem;'>"
                "Learning Roadmap</p>",
                unsafe_allow_html=True,
            )

            sorted_courses = sorted(plan_response.courses, key=lambda c: c.course_order)
            phase_order = {"foundation": 0, "core": 1, "advanced": 2}
            phase_labels = {
                "foundation": "Foundation",
                "core": "Core",
                "advanced": "Advanced",
            }
            phase_colors = {
                "foundation": {
                    "dot": "#3B82F6", "bg": "#DBEAFE",
                    "text": "#1E40AF", "border": "#93C5FD",
                },
                "core": {
                    "dot": "#10B981", "bg": "#D1FAE5",
                    "text": "#065F46", "border": "#6EE7B7",
                },
                "advanced": {
                    "dot": "#8B5CF6", "bg": "#EDE9FE",
                    "text": "#4C1D95", "border": "#C4B5FD",
                },
            }
            default_col = phase_colors["core"]

            phase_counts: dict[str, int] = {}
            for c in sorted_courses:
                p = c.phase.lower()
                phase_counts[p] = phase_counts.get(p, 0) + 1

            tl_css = """
            <style>
            .tl-wrap { position:relative; padding-left:38px; margin-top:4px; }
            .tl-wrap::before {
                content:''; position:absolute; left:14px; top:0; bottom:0;
                width:2px; background:#E2E8F0; border-radius:2px;
            }
            .tl-phase-banner {
                position:relative; margin:1.4rem 0 0.8rem -38px;
                padding:8px 16px; display:inline-flex; align-items:center;
                gap:10px; border-radius:10px; font-family:'Inter',sans-serif;
            }
            .tl-phase-name {
                font-size:0.8rem; font-weight:700;
                letter-spacing:0.06em; text-transform:uppercase;
            }
            .tl-phase-badge {
                font-size:0.72rem; font-weight:600;
                padding:1px 9px; border-radius:10px;
            }
            .tl-card {
                position:relative; margin-bottom:12px;
                background:#fff; border:1px solid #E2E8F0;
                border-radius:12px; padding:1rem 1.2rem;
                box-shadow:0 1px 4px rgba(0,0,0,0.05);
                transition:box-shadow 0.2s, transform 0.2s;
                font-family:'Inter',sans-serif;
            }
            .tl-card:hover {
                box-shadow:0 6px 20px rgba(0,0,0,0.09);
                transform:translateY(-2px);
            }
            .tl-dot {
                position:absolute; left:-30px; top:18px;
                width:16px; height:16px; border-radius:50%;
                border:3px solid #fff;
            }
            .tl-title {
                display:flex; align-items:center; gap:8px;
                font-size:0.97rem; font-weight:700; color:#1E293B; margin-bottom:8px;
            }
            .tl-order {
                display:inline-flex; align-items:center; justify-content:center;
                min-width:22px; height:22px; border-radius:50%;
                font-size:0.72rem; font-weight:700; flex-shrink:0;
            }
            .tl-meta {
                display:flex; gap:14px; flex-wrap:wrap;
                font-size:0.8rem; color:#94A3B8; margin-bottom:8px;
            }
            .tl-skills { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:8px; }
            .tl-skill {
                padding:2px 10px; border-radius:10px;
                font-size:0.75rem; font-weight:500;
            }
            .tl-reasoning {
                font-size:0.83rem; color:#64748B; line-height:1.55;
                border-top:1px solid #F1F5F9; padding-top:8px; margin-top:2px;
            }
            </style>
            """

            html = tl_css + '<div class="tl-wrap fade-in-up">'
            current_phase = None

            for course in sorted_courses:
                phase = course.phase.lower()
                col = phase_colors.get(phase, default_col)

                if phase != current_phase:
                    current_phase = phase
                    label = phase_labels.get(phase, phase.title())
                    count = phase_counts.get(phase, 0)
                    html += (
                        f'<div class="tl-phase-banner" style="background:{col["bg"]};">'
                        f'<span class="tl-phase-name" style="color:{col["text"]};">{label}</span>'
                        f'<span class="tl-phase-badge" '
                        f'style="background:{col["text"]}20;color:{col["text"]};">'
                        f'{count} course{"s" if count != 1 else ""}</span>'
                        f'</div>'
                    )

                skills_html = "".join(
                    f'<span class="tl-skill" '
                    f'style="background:{col["bg"]};color:{col["text"]};">{s}</span>'
                    for s in course.targeted_skills
                )

                html += (
                    f'<div class="tl-card">'
                    f'<div class="tl-dot" '
                    f'style="background:{col["dot"]};box-shadow:0 0 0 3px {col["dot"]}30;"></div>'
                    f'<div class="tl-title">'
                    f'<span class="tl-order" style="background:{col["bg"]};color:{col["text"]};">'
                    f'{course.course_order}</span>'
                    f'{course.course_name}</div>'
                    f'<div class="tl-meta">'
                    f'<span>⏱ {course.estimated_duration_hours:.1f} h</span>'
                    f'<span>📊 {course.targeted_level}</span>'
                    f'<span>🏷 {course.targeted_topic}</span>'
                    f'</div>'
                    f'<div class="tl-skills">{skills_html}</div>'
                    f'<div class="tl-reasoning">{course.reasoning_summary}</div>'
                    f'</div>'
                )

            html += "</div>"
            st.html(html)
