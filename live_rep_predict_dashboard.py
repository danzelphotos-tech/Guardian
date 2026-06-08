"""
Workout Armband — Live Per-Rep Prediction with Dashboard Output
===============================================================
Same as live_rep_predict.py but also writes results to session_log.json
so the Streamlit dashboard can display them in real time.

Usage:
  1. Run this script first:  python live_rep_predict_dashboard.py
  2. Then in a second terminal: streamlit run dashboard.py
"""

import serial
import time
import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
import joblib
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

PORT = "COM13"
BAUD = 115200
MODEL_PATH = "rep_classifier.pkl"
SAMPLE_RATE = 20
SESSION_LOG = "session_log.json"   # Dashboard reads this file

# Rep detection
MIN_REP_TIME = 1.0
MIN_REP_SAMPLES = int(MIN_REP_TIME * SAMPLE_RATE)
SMOOTHING_WINDOW = 11
REP_DETECT_INDEX = 0

BUFFER_SECONDS = 8
BUFFER_SIZE = BUFFER_SECONDS * SAMPLE_RATE

SENSOR_COLS = [
    "accel_x_ms2", "accel_y_ms2", "accel_z_ms2",
    "gyro_x_rads", "gyro_y_rads", "gyro_z_rads",
    "mag_x_uT", "mag_y_uT", "mag_z_uT",
]

SENSOR_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9]


# ============================================================
# Session log — written after every rep
# ============================================================

def init_session_log():
    """Create a fresh session log file at the start of each session."""
    log = {
        "session_start": datetime.now().strftime("%H:%M:%S"),
        "session_active": True,
        "total_reps": 0,
        "good_reps": 0,
        "bad_reps": 0,
        "form_score": 0.0,
        "reps": []
    }
    write_log(log)
    return log


def write_log(log):
    """Write the session log to JSON so the dashboard can read it."""
    with open(SESSION_LOG, "w") as f:
        json.dump(log, f, indent=2)


def update_log(log, prediction, confidence, duration):
    """Add a new rep result to the log and update summary stats."""
    rep_entry = {
        "rep_number": log["total_reps"] + 1,
        "result": prediction,
        "confidence": round(confidence, 1),
        "duration_s": round(duration, 2),
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

    log["reps"].append(rep_entry)
    log["total_reps"] += 1

    if prediction == "good":
        log["good_reps"] += 1
    else:
        log["bad_reps"] += 1

    log["form_score"] = round(log["good_reps"] / log["total_reps"] * 100, 1)
    write_log(log)


# ============================================================
# Feature extraction
# ============================================================

def extract_rep_features(rep_array):
    features = []

    for i, col in enumerate(SENSOR_COLS):
        values = rep_array[:, i]
        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.max(values) - np.min(values),
            np.median(values),
            pd.Series(values).skew(),
            pd.Series(values).kurtosis(),
        ])

    features.append(len(rep_array) / SAMPLE_RATE)

    accel_mag = np.sqrt(rep_array[:, 0]**2 + rep_array[:, 1]**2 + rep_array[:, 2]**2)
    features.append(np.sum(accel_mag**2) / len(rep_array))

    gyro_mag = np.sqrt(rep_array[:, 3]**2 + rep_array[:, 4]**2 + rep_array[:, 5]**2)
    features.append(np.std(gyro_mag))

    return np.array(features)


# ============================================================
# Parse serial line
# ============================================================

def parse_line(line):
    try:
        parts = line.strip().split(",")
        if len(parts) < 11:
            return None
        if "timestamp" in line or "accel" in line:
            return None
        values = [float(parts[i]) for i in SENSOR_INDICES]
        return values
    except (ValueError, IndexError):
        return None


# ============================================================
# Rep detection
# ============================================================

def find_completed_rep(buffer_array):
    if len(buffer_array) < MIN_REP_SAMPLES * 2:
        return None, 0

    signal = buffer_array[:, REP_DETECT_INDEX]

    if len(signal) < SMOOTHING_WINDOW:
        return None, 0
    smoothed = savgol_filter(signal, SMOOTHING_WINDOW, polyorder=3)

    peaks, _ = find_peaks(smoothed, distance=MIN_REP_SAMPLES, prominence=0.3)

    if len(peaks) < 2:
        return None, 0

    start = peaks[-2]
    end = peaks[-1]
    rep_data = buffer_array[start:end]
    rep_duration = len(rep_data) / SAMPLE_RATE

    if 1.0 <= rep_duration <= 6.0:
        return rep_data, end
    else:
        return None, end


# ============================================================
# Main
# ============================================================

def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded!\n")

    print(f"Connecting to {PORT}...")
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("Connected!\n")

    # Initialize fresh session log
    log = init_session_log()

    print("=" * 55)
    print("  LIVE REP-BY-REP PREDICTION + DASHBOARD")
    print("  Open a second terminal and run:")
    print("  streamlit run dashboard.py")
    print("=" * 55)
    print()

    buffer = []

    try:
        while True:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line:
                continue

            values = parse_line(raw_line)
            if values is None:
                continue

            buffer.append(values)

            if len(buffer) > BUFFER_SIZE:
                buffer = buffer[-BUFFER_SIZE:]

            buffer_array = np.array(buffer)
            rep_data, cut_point = find_completed_rep(buffer_array)

            if rep_data is not None:
                features = extract_rep_features(rep_data)
                prediction = model.predict([features])[0]
                confidence = model.predict_proba([features])[0]
                rep_duration = len(rep_data) / SAMPLE_RATE

                if prediction == "good":
                    conf_pct = confidence[1] * 100
                    print(f"  Rep #{log['total_reps']+1:2d}  ✓ GOOD FORM  "
                          f"({conf_pct:.0f}% confidence, {rep_duration:.1f}s)")
                else:
                    conf_pct = confidence[0] * 100
                    print(f"  Rep #{log['total_reps']+1:2d}  ✗ BAD FORM   "
                          f"({conf_pct:.0f}% confidence, {rep_duration:.1f}s)")

                # Update the dashboard log
                update_log(log, prediction, conf_pct, rep_duration)

                buffer = buffer[cut_point:]

    except KeyboardInterrupt:
        log["session_active"] = False
        write_log(log)
        print(f"\n{'='*55}")
        print(f"  SESSION SUMMARY")
        print(f"  Total reps:  {log['total_reps']}")
        print(f"  Good:  {log['good_reps']}  |  Bad: {log['bad_reps']}")
        if log["total_reps"] > 0:
            print(f"  Form score:  {log['form_score']}%")
        print(f"{'='*55}")
        ser.close()


if __name__ == "__main__":
    main()
