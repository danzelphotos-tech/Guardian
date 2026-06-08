"""
Workout Armband — Multi-Exercise Live Prediction (BLE)
======================================================
Based on the original working live_rep_predict_ble.py.
Only changes: loads multiple models, reads exercise from session_log.json.

Requirements:
  pip install bleak
  models/ folder with trained .pkl files (from train_all_exercises.py)

Usage:
  1. Start this script:    python live_rep_predict_ble.py
  2. In second terminal:   streamlit run dashboard.py
  3. Select exercise in the dashboard dropdown
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
import joblib
from datetime import datetime
from bleak import BleakScanner, BleakClient
import os

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE_NAME         = "WorkoutArmband"
CHARACTERISTIC_UUID = "abcdefab-cdef-abcd-efab-cdefabcdefab"
SESSION_LOG         = "session_log.json"
MODEL_DIR           = "models"
SAMPLE_RATE         = 20

# Rep detection
MIN_REP_TIME        = 1.0
MIN_REP_SAMPLES     = int(MIN_REP_TIME * SAMPLE_RATE)
SMOOTHING_WINDOW    = 11
PEAK_PROMINENCE     = 0.3

BUFFER_SECONDS      = 8
BUFFER_SIZE         = BUFFER_SECONDS * SAMPLE_RATE

SENSOR_COLS = [
    "accel_x_ms2", "accel_y_ms2", "accel_z_ms2",
    "gyro_x_rads", "gyro_y_rads", "gyro_z_rads",
    "mag_x_uT",    "mag_y_uT",    "mag_z_uT",
]

SENSOR_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Per-exercise axis config — must match training script
# (axis_index, invert) — axis 0=accel_x, 1=accel_y, 2=accel_z
EXERCISE_AXES = {
    "Bicep Curl":      (2, True),    # accel_z valleys
    "Dumbbell Row":    (1, False),   # accel_y peaks
    "Lat Raise":       (2, False),   # accel_z peaks
    "Shoulder Press":  (0, True),    # accel_x valleys
}

# Exercise models
EXERCISE_MODELS = {
    "Bicep Curl":      "bicep_curl_classifier.pkl",
    "Dumbbell Row":    "dumbbell_row_classifier.pkl",
    "Lat Raise":       "lat_raise_classifier.pkl",
    "Shoulder Press":  "shoulder_press_classifier.pkl",
}

DEFAULT_EXERCISE = "Bicep Curl"


# ============================================================
# LOAD ALL MODELS
# ============================================================

def load_all_models():
    models = {}
    for exercise_name, filename in EXERCISE_MODELS.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            models[exercise_name] = joblib.load(path)
            print(f"  ✓ Loaded {exercise_name} model")
        else:
            print(f"  ✗ Model not found: {path}")
    return models


# ============================================================
# SESSION LOG
# ============================================================

def init_session_log():
    log = {
        "session_start": datetime.now().strftime("%H:%M:%S"),
        "session_active": True,
        "exercise": DEFAULT_EXERCISE,
        "total_reps": 0,
        "good_reps": 0,
        "bad_reps": 0,
        "form_score": 0.0,
        "reps": []
    }
    write_log(log)
    return log


def write_log(log):
    with open(SESSION_LOG, "w") as f:
        json.dump(log, f, indent=2)


def read_log():
    try:
        with open(SESSION_LOG, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def update_log(log, prediction, confidence, duration):
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
# FEATURE EXTRACTION — same as original working version
# ============================================================

def extract_rep_features(rep_array):
    features = []
    for i in range(rep_array.shape[1]):
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
# REP DETECTION — same as original working version
# ============================================================

def find_completed_reps(buffer, axis_index=0, invert=False):
    if len(buffer) < MIN_REP_SAMPLES * 2:
        return None, 0

    buf_array = np.array(buffer)
    signal = buf_array[:, axis_index]

    if len(signal) >= SMOOTHING_WINDOW:
        signal_smooth = savgol_filter(signal, SMOOTHING_WINDOW, polyorder=3)
    else:
        signal_smooth = signal

    if invert:
        signal_smooth = -signal_smooth

    peaks, _ = find_peaks(
        signal_smooth,
        distance=MIN_REP_SAMPLES,
        prominence=PEAK_PROMINENCE
    )

    if len(peaks) < 2:
        return None, 0

    start = peaks[-2]
    end = peaks[-1]
    rep_data = buf_array[start:end]

    return rep_data, end


# ============================================================
# BLE PARSE
# ============================================================

def parse_ble_data(data: bytearray):
    try:
        line = data.decode("utf-8").strip()
        parts = line.split(",")
        if len(parts) < 10:
            return None
        values = [float(parts[i]) for i in SENSOR_INDICES]
        return values
    except (ValueError, IndexError, UnicodeDecodeError):
        return None


# ============================================================
# MAIN
# ============================================================

async def main():
    print("\n" + "=" * 55)
    print("  Smart Workout Armband — Multi-Exercise (BLE)")
    print("=" * 55)

    # Load all models
    print("\nLoading models...")
    models = load_all_models()
    if not models:
        print("ERROR: No models found in models/ folder.")
        print("Run train_all_exercises.py first.")
        return

    # Init session log
    log = init_session_log()
    current_exercise = DEFAULT_EXERCISE
    print(f"\nDefault exercise: {current_exercise}")

    # Scan for device
    print(f"\nScanning for '{DEVICE_NAME}'...")
    device = None
    while device is None:
        devices = await BleakScanner.discover(timeout=5.0)
        for d in devices:
            if d.name == DEVICE_NAME:
                device = d
                break
        if device is None:
            print("  Not found, retrying...")

    print(f"  Found: {device.name} ({device.address})")

    # Connect and stream
    buffer = []
    last_exercise_check = time.time()

    def notification_handler(sender, data):
        nonlocal buffer, log, current_exercise, last_exercise_check

        values = parse_ble_data(data)
        if values is None:
            return

        buffer.append(values)

        # Keep buffer from growing forever
        if len(buffer) > BUFFER_SIZE * 2:
            buffer = buffer[-BUFFER_SIZE:]

        # Check if dashboard changed the exercise (once per second)
        now = time.time()
        if now - last_exercise_check > 1.0:
            last_exercise_check = now
            try:
                current_log = read_log()
                if current_log and current_log.get("exercise") != current_exercise:
                    new_exercise = current_log.get("exercise", DEFAULT_EXERCISE)
                    if new_exercise in models:
                        current_exercise = new_exercise
                        log = current_log
                        log["total_reps"] = 0
                        log["good_reps"] = 0
                        log["bad_reps"] = 0
                        log["form_score"] = 0.0
                        log["reps"] = []
                        write_log(log)
                        buffer.clear()
                        print(f"\n  >>> Switched to: {current_exercise}")
            except Exception as e:
                print(f"  LOG ERROR: {e}")

        # Try to find a completed rep
        try:
            axis_index, invert = EXERCISE_AXES.get(current_exercise, (0, False))
            rep_data, cut_point = find_completed_reps(buffer, axis_index=axis_index, invert=invert)
            if rep_data is not None and current_exercise in models:
                model = models[current_exercise]
                features = extract_rep_features(rep_data).reshape(1, -1)
                prediction = model.predict(features)[0]
                confidence = model.predict_proba(features)[0]
                rep_duration = len(rep_data) / SAMPLE_RATE

                # Model returns "good"/"bad" strings (trained with string labels)
                result = "good" if prediction == "good" else "bad"

                if result == "good":
                    conf_pct = confidence[1] * 100
                    print(f"  [{current_exercise}] Rep #{log['total_reps']+1:2d}  "
                          f"✓ GOOD FORM  ({conf_pct:.0f}%, {rep_duration:.1f}s)")
                else:
                    conf_pct = confidence[0] * 100
                    print(f"  [{current_exercise}] Rep #{log['total_reps']+1:2d}  "
                          f"✗ BAD FORM   ({conf_pct:.0f}%, {rep_duration:.1f}s)")

                update_log(log, result, conf_pct, rep_duration)
                buffer[:] = buffer[cut_point:]

        except Exception as e:
            print(f"  REP ERROR: {e}")

    async with BleakClient(device.address) as client:
        print(f"  Connected! Streaming data...")
        print(f"  Select exercise in the dashboard dropdown.")
        print(f"  Press Ctrl+C to stop.\n")

        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

        try:
            while True:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            log["session_active"] = False
            write_log(log)
            print(f"\n{'='*55}")
            print(f"  SESSION SUMMARY — {current_exercise}")
            print(f"  Total reps:  {log['total_reps']}")
            print(f"  Good: {log['good_reps']}  |  Bad: {log['bad_reps']}")
            if log["total_reps"] > 0:
                print(f"  Form score:  {log['form_score']}%")
            print(f"{'='*55}")


if __name__ == "__main__":
    asyncio.run(main())