"""
Workout Armband — Live Per-Rep Prediction with BLE + Dashboard
==============================================================
Wireless version of live_rep_predict_dashboard.py.
Reads IMU data over BLE instead of USB serial.

Requirements:
  pip install bleak
  rep_classifier.pkl must be in the same folder

Usage:
  1. Run this script:      python live_rep_predict_ble.py
  2. In second terminal:   streamlit run dashboard.py
"""

import asyncio
import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
import joblib
from datetime import datetime
from bleak import BleakScanner, BleakClient

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE_NAME         = "WorkoutArmband"
CHARACTERISTIC_UUID = "abcdefab-cdef-abcd-efab-cdefabcdefab"
MODEL_PATH          = "rep_classifier.pkl"
SESSION_LOG         = "session_log.json"
SAMPLE_RATE         = 20

# Rep detection — must match training settings
MIN_REP_TIME        = 1.0
MIN_REP_SAMPLES     = int(MIN_REP_TIME * SAMPLE_RATE)
SMOOTHING_WINDOW    = 11
REP_DETECT_INDEX    = 0       # accel_x is first sensor column

BUFFER_SECONDS      = 8
BUFFER_SIZE         = BUFFER_SECONDS * SAMPLE_RATE

SENSOR_COLS = [
    "accel_x_ms2", "accel_y_ms2", "accel_z_ms2",
    "gyro_x_rads", "gyro_y_rads", "gyro_z_rads",
    "mag_x_uT",    "mag_y_uT",    "mag_z_uT",
]

# Sensor column indices in the CSV line from Arduino
# elapsed_ms(0), accel_x(1), accel_y(2), accel_z(3),
# gyro_x(4), gyro_y(5), gyro_z(6), mag_x(7), mag_y(8), mag_z(9), temp(10)
SENSOR_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9]


# ============================================================
# SESSION LOG
# ============================================================

def init_session_log():
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
    with open(SESSION_LOG, "w") as f:
        json.dump(log, f, indent=2)


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
# FEATURE EXTRACTION
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
# PARSE BLE DATA LINE
# ============================================================

def parse_line(line):
    try:
        parts = line.strip().split(",")
        if len(parts) < 11:
            return None
        values = [float(parts[i]) for i in SENSOR_INDICES]
        return values
    except (ValueError, IndexError):
        return None


# ============================================================
# REP DETECTION
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
    return None, end


# ============================================================
# MAIN — BLE async loop
# ============================================================

async def main():
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded!\n")

    # Scan for device
    print(f"Scanning for '{DEVICE_NAME}'...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10)

    if device is None:
        print(f"Could not find '{DEVICE_NAME}'. Make sure:")
        print("  1. The BLE Arduino sketch is uploaded and running")
        print("  2. The ESP32 is powered on")
        print("  3. No other device is already connected to it")
        return

    print(f"Found: {device.name} ({device.address})")

    # Initialize session log
    log = init_session_log()

    # Shared buffer between BLE callback and rep detection
    buffer = []

    print("=" * 55)
    print("  LIVE REP-BY-REP PREDICTION — BLE MODE")
    print("  Open a second terminal and run:")
    print("  streamlit run dashboard.py")
    print("=" * 55)
    print()

    def handle_notification(sender, data):
        """Called every time a new BLE data packet arrives."""
        line = data.decode("utf-8").strip()
        values = parse_line(line)
        if values is None:
            return

        buffer.append(values)

        # Keep buffer from growing forever
        if len(buffer) > BUFFER_SIZE:
            buffer.pop(0)

        if len(buffer) < MIN_REP_SAMPLES * 2:
            return

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

            update_log(log, prediction, conf_pct, rep_duration)

            # Trim buffer to the last peak
            del buffer[:cut_point]

    # Connect and subscribe
    async with BleakClient(device) as client:
        print(f"Connected to {device.name}! Starting predictions...\n")
        await client.start_notify(CHARACTERISTIC_UUID, handle_notification)

        try:
            while True:
                await asyncio.sleep(0.1)
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
            await client.stop_notify(CHARACTERISTIC_UUID)


if __name__ == "__main__":
    asyncio.run(main())
