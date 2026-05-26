"""
Workout Armband — Live Dashboard
=================================
Run this in a second terminal while live_rep_predict_dashboard.py is running:
  streamlit run dashboard.py
"""

import streamlit as st
import json
import time
import os
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Workout Armband",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Dark background */
.stApp {
    background-color: #0D0D0F;
    color: #E8E8E8;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Main title */
.main-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    letter-spacing: 0.1em;
    color: #FFFFFF;
    margin: 0;
    padding: 0;
    line-height: 1;
}

.subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #666;
    margin-top: 4px;
}

/* Stat cards */
.stat-card {
    background: #161618;
    border: 1px solid #2A2A2E;
    border-radius: 12px;
    padding: 24px 28px;
    text-align: center;
}

.stat-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    line-height: 1;
    margin: 0;
}

.stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    margin-top: 6px;
}

/* Last rep result */
.result-good {
    background: linear-gradient(135deg, #0D2E1A, #0F3D20);
    border: 1px solid #1A6B35;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.result-bad {
    background: linear-gradient(135deg, #2E0D0D, #3D0F0F);
    border: 1px solid #6B1A1A;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.result-waiting {
    background: #161618;
    border: 1px solid #2A2A2E;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.result-icon {
    font-size: 3.5rem;
    line-height: 1;
    margin-bottom: 8px;
}

.result-text {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    letter-spacing: 0.08em;
    margin: 0;
}

.result-good .result-text { color: #3DDC73; }
.result-bad .result-text  { color: #DC3D3D; }
.result-waiting .result-text { color: #555; }

.result-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: #888;
    margin-top: 6px;
    letter-spacing: 0.05em;
}

/* Form score bar */
.score-bar-container {
    background: #1E1E22;
    border-radius: 100px;
    height: 10px;
    margin-top: 8px;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.5s ease;
}

/* Session status dot */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
}

.status-active { background: #3DDC73; }
.status-inactive { background: #555; animation: none; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Table styling */
.rep-table {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    width: 100%;
    border-collapse: collapse;
}

.rep-table th {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #555;
    padding: 8px 12px;
    border-bottom: 1px solid #222;
    text-align: left;
}

.rep-table td {
    padding: 10px 12px;
    border-bottom: 1px solid #1A1A1E;
    color: #CCC;
}

.badge-good {
    background: #0F3D20;
    color: #3DDC73;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.badge-bad {
    background: #3D0F0F;
    color: #DC3D3D;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* Section headers */
.section-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 12px;
    margin-top: 28px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD SESSION DATA
# ============================================================

def load_session():
    """Load the session log JSON. Returns default if not found."""
    if not os.path.exists("session_log.json"):
        return {
            "session_start": "--:--:--",
            "session_active": False,
            "total_reps": 0,
            "good_reps": 0,
            "bad_reps": 0,
            "form_score": 0.0,
            "reps": []
        }
    try:
        with open("session_log.json", "r") as f:
            return json.load(f)
    except:
        return None


# ============================================================
# DASHBOARD LAYOUT
# ============================================================

# Header
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown('<p class="main-title">WORKOUT ARMBAND</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Bicep Curl Form Monitor</p>', unsafe_allow_html=True)

# Auto-refresh placeholder
refresh_placeholder = st.empty()

# Load data
data = load_session()

if data is None:
    st.error("Error reading session log.")
    st.stop()

# Session status
with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    if data["session_active"]:
        st.markdown(
            f'<p style="font-family: DM Sans; font-size: 0.8rem; color: #3DDC73; text-align: right;">'
            f'<span class="status-dot status-active">●</span> LIVE</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<p style="font-family: DM Sans; font-size: 0.8rem; color: #555; text-align: right;">'
            f'<span class="status-dot status-inactive">●</span> IDLE — start recording</p>',
            unsafe_allow_html=True
        )

st.markdown("---")

# ============================================================
# TOP ROW — Stats + Last Rep
# ============================================================

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 0.3, 2])

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <p class="stat-value" style="color: #FFFFFF;">{data['total_reps']}</p>
        <p class="stat-label">Total Reps</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <p class="stat-value" style="color: #3DDC73;">{data['good_reps']}</p>
        <p class="stat-label">Good Form</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <p class="stat-value" style="color: #DC3D3D;">{data['bad_reps']}</p>
        <p class="stat-label">Bad Form</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    # Last rep result
    if data["total_reps"] == 0:
        st.markdown("""
        <div class="result-waiting">
            <p class="result-icon">🏋️</p>
            <p class="result-text">WAITING FOR REPS</p>
            <p class="result-sub">Start curling to see live feedback</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        last = data["reps"][-1]
        if last["result"] == "good":
            st.markdown(f"""
            <div class="result-good">
                <p class="result-icon">✓</p>
                <p class="result-text">GOOD FORM</p>
                <p class="result-sub">Rep #{last['rep_number']} — {last['confidence']:.0f}% confidence — {last['duration_s']}s</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-bad">
                <p class="result-icon">✗</p>
                <p class="result-text">BAD FORM</p>
                <p class="result-sub">Rep #{last['rep_number']} — {last['confidence']:.0f}% confidence — {last['duration_s']}s</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# FORM SCORE BAR
# ============================================================

st.markdown('<p class="section-header">Form Score</p>', unsafe_allow_html=True)

score = data["form_score"]
score_color = "#3DDC73" if score >= 70 else "#F0A500" if score >= 50 else "#DC3D3D"

col_score, col_pct = st.columns([5, 1])
with col_score:
    st.markdown(f"""
    <div class="score-bar-container">
        <div class="score-bar-fill" style="width: {score}%; background: {score_color};"></div>
    </div>
    """, unsafe_allow_html=True)
with col_pct:
    st.markdown(f"""
    <p style="font-family: Bebas Neue; font-size: 1.4rem; color: {score_color}; 
       margin: 0; text-align: right; line-height: 1.2;">{score:.0f}%</p>
    """, unsafe_allow_html=True)

# ============================================================
# CHARTS + REP HISTORY
# ============================================================

if data["total_reps"] > 0:
    col_chart, col_table = st.columns([1.5, 1])

    reps_df = pd.DataFrame(data["reps"])

    with col_chart:
        st.markdown('<p class="section-header">Rep Duration Over Time</p>', unsafe_allow_html=True)

        # Color code by result
        chart_data = reps_df[["rep_number", "duration_s", "result"]].copy()
        chart_data["color"] = chart_data["result"].map({"good": "#3DDC73", "bad": "#DC3D3D"})

        st.line_chart(
            chart_data.set_index("rep_number")["duration_s"],
            color="#5B8DEF",
            use_container_width=True
        )

    with col_table:
        st.markdown('<p class="section-header">Rep History</p>', unsafe_allow_html=True)

        # Build HTML table — most recent reps first
        rows_html = ""
        for _, row in reps_df.iloc[::-1].head(15).iterrows():
            badge = (f'<span class="badge-good">GOOD</span>'
                     if row["result"] == "good"
                     else f'<span class="badge-bad">BAD</span>')
            rows_html += f"""
            <tr>
                <td>#{int(row['rep_number'])}</td>
                <td>{badge}</td>
                <td>{row['confidence']:.0f}%</td>
                <td>{row['duration_s']}s</td>
                <td style="color:#555">{row['timestamp']}</td>
            </tr>
            """

        st.markdown(f"""
        <table class="rep-table">
            <thead>
                <tr>
                    <th>Rep</th>
                    <th>Result</th>
                    <th>Confidence</th>
                    <th>Duration</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """, unsafe_allow_html=True)

# ============================================================
# AUTO REFRESH
# ============================================================

# Refresh every 1 second while session is active
if data["session_active"]:
    time.sleep(1)
    st.rerun()
else:
    st.markdown("""
    <p style="font-family: DM Sans; font-size: 0.8rem; color: #444; text-align: center; margin-top: 40px;">
    Run live_rep_predict_dashboard.py to start a session
    </p>
    """, unsafe_allow_html=True)
