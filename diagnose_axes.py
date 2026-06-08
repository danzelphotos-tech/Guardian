"""
Workout Armband — Axis Diagnostic
==================================
Checks each exercise's data to find which accelerometer axis
and direction (peaks vs valleys) gives the cleanest rep detection.

Usage:
  python diagnose_axes.py
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter

DATA_DIR = "data"
SAMPLE_RATE = 20
MIN_REP_TIME = 1.0
MIN_REP_SAMPLES = int(MIN_REP_TIME * SAMPLE_RATE)
SMOOTHING_WINDOW = 11
TRIM_START = 5 * SAMPLE_RATE
TRIM_END = 2 * SAMPLE_RATE

EXERCISES = ["Bicep_curl", "Dumbbell Row", "Lat_raise", "Shoulder_press"]
AXES = ["accel_x_ms2", "accel_y_ms2", "accel_z_ms2"]


def test_axis(df, axis, invert=False):
    """Try finding reps on a given axis. Returns count of valid reps."""
    if axis not in df.columns:
        return 0, 0

    signal = df[axis].values
    if len(signal) < SMOOTHING_WINDOW:
        return 0, 0

    smoothed = savgol_filter(signal, SMOOTHING_WINDOW, polyorder=3)
    if invert:
        smoothed = -smoothed

    peaks, _ = find_peaks(smoothed, distance=MIN_REP_SAMPLES, prominence=0.3)

    # Count reps with reasonable duration (1-6s)
    valid_reps = 0
    for i in range(len(peaks) - 1):
        dur = (peaks[i+1] - peaks[i]) / SAMPLE_RATE
        if 1.0 <= dur <= 6.0:
            valid_reps += 1

    return len(peaks), valid_reps


def diagnose_exercise(exercise_name, exercise_dir):
    """Test all axes on a few sessions and report results."""
    print(f"\n{'='*60}")
    print(f"  {exercise_name}")
    print(f"{'='*60}")

    # Load first 5 good sessions
    good_dir = os.path.join(exercise_dir, "good")
    if not os.path.exists(good_dir):
        print(f"  No good/ folder found")
        return

    csvs = sorted(glob.glob(os.path.join(good_dir, "*.csv")))[:5]
    if not csvs:
        print(f"  No CSVs found in good/")
        return

    # Test each axis on each session
    results = {}
    for axis in AXES:
        for direction in ["peaks", "valleys"]:
            key = f"{axis} ({direction})"
            results[key] = {"total_peaks": 0, "valid_reps": 0, "sessions": 0}

    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
            df = df.drop(columns=["flex_raw", "flex_voltage_v"], errors="ignore")

            if len(df) > TRIM_START + TRIM_END + MIN_REP_SAMPLES:
                df = df.iloc[TRIM_START:-TRIM_END].reset_index(drop=True)

            fname = os.path.basename(csv_path)

            for axis in AXES:
                for direction, invert in [("peaks", False), ("valleys", True)]:
                    key = f"{axis} ({direction})"
                    total, valid = test_axis(df, axis, invert)
                    results[key]["total_peaks"] += total
                    results[key]["valid_reps"] += valid
                    results[key]["sessions"] += 1

        except Exception as e:
            print(f"  ERROR: {os.path.basename(csv_path)}: {e}")

    # Report
    print(f"\n  {'Axis':<30} {'Peaks':>8} {'Valid Reps':>12} {'Avg/Session':>12}")
    print(f"  {'-'*62}")

    best_key = None
    best_reps = 0

    for key, data in sorted(results.items(), key=lambda x: -x[1]["valid_reps"]):
        avg = data["valid_reps"] / max(data["sessions"], 1)
        marker = ""
        if data["valid_reps"] > best_reps:
            best_reps = data["valid_reps"]
            best_key = key
        print(f"  {key:<30} {data['total_peaks']:>8} {data['valid_reps']:>12} {avg:>12.1f}")

    print(f"\n  >>> BEST: {best_key} ({best_reps} valid reps)")
    return best_key


def main():
    print("Workout Armband — Axis Diagnostic")
    print("Testing which axis gives the best rep detection per exercise\n")

    for exercise in EXERCISES:
        exercise_dir = os.path.join(DATA_DIR, exercise)
        if os.path.exists(exercise_dir):
            diagnose_exercise(exercise, exercise_dir)
        else:
            print(f"\n  SKIP: {exercise} — folder not found")

    print(f"\n{'='*60}")
    print("  Use these results to set the axis per exercise in")
    print("  train_all_exercises.py and live_rep_predict_ble.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
