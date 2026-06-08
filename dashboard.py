"""
Workout Armband — Live Dashboard (Multi-Exercise)
==================================================
Run this in a second terminal while live_rep_predict_ble.py is running:
  streamlit run dashboard.py

The exercise dropdown controls which model the BLE script uses.
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
    initial_sidebar_state="expanded"
)

SESSION_LOG = "session_log.json"
EXERCISES = ["Bicep Curl", "Dumbbell Row", "Lat Raise", "Shoulder Press"]


# ============================================================
# READ / WRITE SESSION LOG
# ============================================================

def read_log():
    """Read the session log written by the BLE prediction script."""
    if not os.path.exists(SESSION_LOG):
        return None
    try:
        with open(SESSION_LOG, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def write_exercise_selection(exercise):
    """Update just the exercise field in the session log."""
    log = read_log()
    if log is None:
        log = {
            "session_start": "",
            "session_active": False,
            "exercise": exercise,
            "total_reps": 0,
            "good_reps": 0,
            "bad_reps": 0,
            "form_score": 0.0,
            "reps": []
        }
    log["exercise"] = exercise
    with open(SESSION_LOG, "w") as f:
        json.dump(log, f, indent=2)


# ============================================================
# SIDEBAR — Exercise Selection
# ============================================================

st.sidebar.title("⚙️ Settings")

# Get current exercise from log (or default)
current_log = read_log()
current_exercise = current_log.get("exercise", "Bicep Curl") if current_log else "Bicep Curl"
default_index = EXERCISES.index(current_exercise) if current_exercise in EXERCISES else 0

selected_exercise = st.sidebar.selectbox(
    "Exercise",
    EXERCISES,
    index=default_index,
    help="Switching resets your rep count for the new exercise"
)

# If exercise changed, write it to the log so the BLE script picks it up
if selected_exercise != current_exercise:
    write_exercise_selection(selected_exercise)
    st.sidebar.success(f"Switched to {selected_exercise}")
    time.sleep(0.3)
    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("💪 Smart Workout Armband")
st.caption(f"Exercise: **{selected_exercise}** · Dashboard auto-refreshes every second")

# ============================================================
# MAIN DISPLAY
# ============================================================

log = read_log()

if log is None or not log.get("session_active", False):
    st.info(
        "Waiting for session to start...\n\n"
        "1. Run `python live_rep_predict_ble.py` in a terminal\n"
        "2. Select your exercise above\n"
        "3. Start exercising!"
    )
else:
    # ---- Summary metrics ----
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Reps", log.get("total_reps", 0))
    with col2:
        st.metric("Good Reps", log.get("good_reps", 0))
    with col3:
        st.metric("Bad Reps", log.get("bad_reps", 0))
    with col4:
        score = log.get("form_score", 0)
        st.metric("Form Score", f"{score}%")

    st.divider()

    # ---- Last rep verdict ----
    reps = log.get("reps", [])
    if reps:
        last = reps[-1]
        result = last.get("result", "")
        conf = last.get("confidence", 0)
        dur = last.get("duration_s", 0)

        if result == "good":
            st.success(
                f"**Rep #{last['rep_number']}:  ✓ GOOD FORM**  —  "
                f"{conf:.0f}% confidence  ·  {dur:.1f}s"
            )
        else:
            st.error(
                f"**Rep #{last['rep_number']}:  ✗ BAD FORM**  —  "
                f"{conf:.0f}% confidence  ·  {dur:.1f}s"
            )

        # ---- Rep history table ----
        st.subheader("Rep History")
        df = pd.DataFrame(reps)
        df = df[["rep_number", "result", "confidence", "duration_s", "timestamp"]]
        df.columns = ["Rep #", "Result", "Confidence %", "Duration (s)", "Time"]

        # Color the Result column
        def color_result(val):
            if val == "good":
                return "background-color: #d4edda"
            return "background-color: #f8d7da"

        st.dataframe(
            df.style.map(color_result, subset=["Result"]),
            use_container_width=True,
            hide_index=True
        )

        # ---- Duration chart ----
        st.subheader("Rep Duration")
        chart_df = pd.DataFrame({
            "Rep": [r["rep_number"] for r in reps],
            "Duration (s)": [r["duration_s"] for r in reps],
        })
        st.bar_chart(chart_df.set_index("Rep"))

    else:
        st.info("No reps detected yet — start exercising!")


# ============================================================
# AUTO-REFRESH
# ============================================================

time.sleep(1)
st.rerun()
