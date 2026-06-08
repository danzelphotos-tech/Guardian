"""
Workout Armband — Multi-Exercise Model Trainer
===============================================
Trains a separate Random Forest classifier for each exercise.
Based on the original train_rep_classifier.py, extended for multiple exercises.

Each exercise folder must contain good/ and bad/ subfolders with CSV files.

Folder structure expected:
  workout-armband/
    data/
      Bicep_curl/
        good/   <- good form CSVs
        bad/    <- bad form CSVs
      Dumbbell Row/
        good/
        bad/
      Lat_raise/
        good/
        bad/
      Shoulder_press/
        good/
        bad/
    train_all_exercises.py  (this script)

Output:
  models/
    bicep_curl_classifier.pkl
    dumbbell_row_classifier.pkl
    lat_raise_classifier.pkl
    shoulder_press_classifier.pkl

Usage:
  python train_all_exercises.py
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
MODEL_DIR = "models"
TEST_SIZE = 0.2
RANDOM_SEED = 42
SAMPLE_RATE = 20              # 20Hz from Arduino sketch

# Trim noisy start and end of each session
TRIM_START_SECONDS = 5
TRIM_END_SECONDS = 2

# Rep detection settings
MIN_REP_TIME = 1.0            # Minimum seconds between reps
MIN_REP_SAMPLES = int(MIN_REP_TIME * SAMPLE_RATE)
SMOOTHING_WINDOW = 11         # Savitzky-Golay window (must be odd)

# Which accelerometer axis to use for rep detection — must match live script
REP_DETECT_COL = "accel_x_ms2"

# Per-exercise axis config (from diagnose_axes.py results)
# Each entry: (column_name, invert) — invert=True means detect valleys
EXERCISE_AXES = {
    "Bicep_curl":      ("accel_z_ms2", True),    # accel_z valleys
    "Dumbbell Row":    ("accel_y_ms2", False),   # accel_y peaks
    "Lat_raise":       ("accel_z_ms2", False),   # accel_z peaks
    "Shoulder_press":  ("accel_x_ms2", True),    # accel_x valleys
}

# All sensor columns for feature extraction
SENSOR_COLS = [
    "accel_x_ms2", "accel_y_ms2", "accel_z_ms2",
    "gyro_x_rads", "gyro_y_rads", "gyro_z_rads",
    "mag_x_uT", "mag_y_uT", "mag_z_uT",
]

# Exercise folder names → model file names
EXERCISES = {
    "Bicep_curl":      "bicep_curl_classifier.pkl",
    "Dumbbell Row":    "dumbbell_row_classifier.pkl",
    "Lat_raise":       "lat_raise_classifier.pkl",
    "Shoulder_press":  "shoulder_press_classifier.pkl",
}


# ============================================================
# STEP 1 — Load all CSVs for one exercise
# ============================================================

def load_exercise_data(exercise_dir):
    """Walk through good/ and bad/ folders, load every CSV, tag with label."""
    all_dataframes = []

    trim_start_rows = TRIM_START_SECONDS * SAMPLE_RATE
    trim_end_rows = TRIM_END_SECONDS * SAMPLE_RATE

    for label in ["good", "bad"]:
        folder = os.path.join(exercise_dir, label)
        if not os.path.exists(folder):
            print(f"    WARNING: {folder} not found, skipping")
            continue

        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        print(f"    {label}: {len(csv_files)} sessions")

        for filepath in csv_files:
            try:
                df = pd.read_csv(filepath)

                # Trim noisy start and end
                if len(df) > (trim_start_rows + trim_end_rows + MIN_REP_SAMPLES):
                    df = df.iloc[trim_start_rows:-trim_end_rows].reset_index(drop=True)

                # Drop flex sensor columns if present (old hardware version)
                df = df.drop(columns=["flex_raw", "flex_voltage_v"], errors="ignore")

                df["label"] = label
                df["source_file"] = os.path.basename(filepath)
                all_dataframes.append(df)

            except Exception as e:
                print(f"    ERROR loading {os.path.basename(filepath)}: {e}")

    if not all_dataframes:
        return None

    combined = pd.concat(all_dataframes, ignore_index=True)
    print(f"    Total rows after trimming: {len(combined):,}")
    return combined


# ============================================================
# STEP 2 — Detect individual reps in a session
# ============================================================

def detect_reps(session_df, axis_col=REP_DETECT_COL, invert=False):
    """
    Find individual reps by detecting peaks in the specified axis.
    If invert=True, detects valleys instead of peaks.
    """
    signal = session_df[axis_col].values

    if len(signal) < SMOOTHING_WINDOW:
        return []

    smoothed = savgol_filter(signal, SMOOTHING_WINDOW, polyorder=3)

    if invert:
        smoothed = -smoothed

    peaks, _ = find_peaks(
        smoothed,
        distance=MIN_REP_SAMPLES,
        prominence=0.3
    )

    # Each rep = data from one peak to the next
    reps = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        rep_data = session_df.iloc[start:end].reset_index(drop=True)

        rep_duration = len(rep_data) / SAMPLE_RATE
        if 1.0 <= rep_duration <= 6.0:
            reps.append(rep_data)

    return reps


# ============================================================
# STEP 3 — Extract features from a single rep
# ============================================================

def extract_rep_features(rep_df):
    """
    Summarize one complete rep into a feature vector.
    Same features as the original train_rep_classifier.py.
    """
    features = {}

    for col in SENSOR_COLS:
        values = rep_df[col].values

        # Basic statistics
        features[f"{col}_mean"] = np.mean(values)
        features[f"{col}_std"] = np.std(values)
        features[f"{col}_min"] = np.min(values)
        features[f"{col}_max"] = np.max(values)
        features[f"{col}_range"] = np.max(values) - np.min(values)

        # Shape of the movement
        features[f"{col}_median"] = np.median(values)
        features[f"{col}_skew"] = pd.Series(values).skew()
        features[f"{col}_kurtosis"] = pd.Series(values).kurtosis()

    # Rep duration in seconds
    features["rep_duration_s"] = len(rep_df) / SAMPLE_RATE

    # Total movement energy
    accel_magnitude = np.sqrt(
        rep_df["accel_x_ms2"].values**2 +
        rep_df["accel_y_ms2"].values**2 +
        rep_df["accel_z_ms2"].values**2
    )
    features["accel_energy"] = np.sum(accel_magnitude**2) / len(rep_df)

    # Smoothness — std of gyroscope magnitude
    gyro_magnitude = np.sqrt(
        rep_df["gyro_x_rads"].values**2 +
        rep_df["gyro_y_rads"].values**2 +
        rep_df["gyro_z_rads"].values**2
    )
    features["gyro_smoothness"] = np.std(gyro_magnitude)

    return features


# ============================================================
# STEP 4 — Build the per-rep dataset for one exercise
# ============================================================

def build_rep_dataset(combined_df, axis_col=REP_DETECT_COL, invert=False):
    """
    Go through each session, detect reps, extract features per rep,
    and label each rep based on the session label.
    """
    all_rep_features = []
    total_reps = 0
    good_reps = 0
    bad_reps = 0

    for source_file, session in combined_df.groupby("source_file"):
        session = session.reset_index(drop=True)
        label = session["label"].iloc[0]

        reps = detect_reps(session, axis_col=axis_col, invert=invert)

        for rep_df in reps:
            features = extract_rep_features(rep_df)
            features["label"] = label
            all_rep_features.append(features)
            total_reps += 1
            if label == "good":
                good_reps += 1
            else:
                bad_reps += 1

        print(f"    {source_file}: {len(reps)} reps ({label})")

    if not all_rep_features:
        return None

    rep_dataset = pd.DataFrame(all_rep_features)
    print(f"\n    Total reps: {total_reps} (good: {good_reps}, bad: {bad_reps})")
    return rep_dataset


# ============================================================
# STEP 5 — Train one exercise model
# ============================================================

def train_exercise(exercise_name, exercise_dir, model_path):
    """Full pipeline: load → detect reps → extract features → train → save."""
    print(f"\n{'='*60}")
    print(f"  Training: {exercise_name}")
    print(f"{'='*60}")

    # Load data
    combined = load_exercise_data(exercise_dir)
    if combined is None:
        print(f"  ERROR: No data loaded for {exercise_name}")
        return False

    # Look up per-exercise axis config
    axis_col, invert = EXERCISE_AXES.get(exercise_name, (REP_DETECT_COL, False))
    direction = "valleys" if invert else "peaks"
    print(f"    Rep detection: {axis_col} ({direction})")

    # Detect reps and extract features
    print(f"    Detecting reps and extracting features...")
    rep_dataset = build_rep_dataset(combined, axis_col=axis_col, invert=invert)
    if rep_dataset is None or len(rep_dataset) < 10:
        count = 0 if rep_dataset is None else len(rep_dataset)
        print(f"  ERROR: Only {count} reps found — need at least 10")
        return False

    # Train/test split
    X = rep_dataset.drop("label", axis=1)
    y = rep_dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n    Train: {len(X_train)} reps  |  Test: {len(X_test)} reps")

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"\n    Test accuracy: {acc*100:.1f}%")
    print(f"    Confusion matrix:\n{confusion_matrix(y_test, predictions)}")
    print(f"\n{classification_report(y_test, predictions)}")

    # Save model
    joblib.dump(model, model_path)
    print(f"    Model saved: {model_path}")

    return True


# ============================================================
# MAIN — Train all exercises
# ============================================================

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    results = {}

    for exercise_folder, model_filename in EXERCISES.items():
        exercise_dir = os.path.join(DATA_DIR, exercise_folder)
        model_path = os.path.join(MODEL_DIR, model_filename)

        if not os.path.exists(exercise_dir):
            print(f"\n  SKIPPING {exercise_folder} — folder not found at {exercise_dir}")
            results[exercise_folder] = "SKIPPED"
            continue

        success = train_exercise(exercise_folder, exercise_dir, model_path)
        results[exercise_folder] = "OK" if success else "FAILED"

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*60}")
    for exercise, status in results.items():
        icon = "✓" if status == "OK" else "✗" if status == "FAILED" else "—"
        print(f"  {icon}  {exercise}: {status}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()