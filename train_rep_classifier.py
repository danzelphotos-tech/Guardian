"""
Workout Armband — Per-Rep Bicep Curl Classifier
================================================
Detects individual reps from IMU data, extracts features from each rep,
and trains a Random Forest to classify each rep as good or bad form.

Stage 1: Rep detection — finds peaks in accelerometer data to segment reps
Stage 2: Per-rep feature extraction and classification

Folder structure expected:
  workout-armband/
    data/
      good/   <- good form CSV files
      bad/    <- bad form CSV files
    train_rep_classifier.py  (this script)
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
TEST_SIZE = 0.2
RANDOM_SEED = 42
SAMPLE_RATE = 20              # 20Hz from Arduino sketch

# Trim noisy start and end of each session
TRIM_START_SECONDS = 5
TRIM_END_SECONDS = 2

# Rep detection settings
MIN_REP_TIME = 1.0            # Minimum seconds between reps (no one curls faster than this)
MIN_REP_SAMPLES = int(MIN_REP_TIME * SAMPLE_RATE)  # Convert to rows
SMOOTHING_WINDOW = 11         # Window size for smoothing filter (must be odd)

# Which accelerometer axis to use for rep detection
# accel_x_ms2 showed the clearest rep pattern in your data (2.4 → -1.5 per curl)
REP_DETECT_COL = "accel_x_ms2"

# All sensor columns for feature extraction
SENSOR_COLS = [
    "accel_x_ms2", "accel_y_ms2", "accel_z_ms2",
    "gyro_x_rads", "gyro_y_rads", "gyro_z_rads",
    "mag_x_uT", "mag_y_uT", "mag_z_uT",
]


# ============================================================
# STEP 1 — Load all CSVs and tag with labels
# ============================================================

def load_all_data():
    """Walk through good/ and bad/ folders, load every CSV, tag with label."""
    all_dataframes = []

    trim_start_rows = TRIM_START_SECONDS * SAMPLE_RATE
    trim_end_rows = TRIM_END_SECONDS * SAMPLE_RATE

    print(f"Trimming first {TRIM_START_SECONDS}s and last {TRIM_END_SECONDS}s of each session")

    for label in ["good", "bad"]:
        folder = os.path.join(DATA_DIR, label)
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        print(f"Found {len(csv_files)} files in {folder}/")

        for filepath in csv_files:
            df = pd.read_csv(filepath)

            # Trim noisy start and end
            if len(df) > (trim_start_rows + trim_end_rows + MIN_REP_SAMPLES):
                df = df.iloc[trim_start_rows:-trim_end_rows].reset_index(drop=True)

            # Drop flex sensor columns if present
            df = df.drop(columns=["flex_raw", "flex_voltage_v"], errors="ignore")

            df["label"] = label
            df["source_file"] = os.path.basename(filepath)
            all_dataframes.append(df)

    combined = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total rows after trimming: {len(combined):,}")
    return combined


# ============================================================
# STEP 2 — Detect individual reps in a session
# ============================================================

def detect_reps(session_df):
    """
    Find individual reps by detecting peaks in the accelerometer signal.
    Each peak corresponds to the "extended arm" position (top of accel_x).
    The data between consecutive peaks = one complete rep.
    
    Returns a list of DataFrames, one per rep.
    """
    signal = session_df[REP_DETECT_COL].values

    # Smooth the signal to remove noise — Savitzky-Golay filter
    # (SG filter = Savitzky-Golay, a smoothing method that preserves
    # the shape of peaks better than a simple moving average)
    if len(signal) < SMOOTHING_WINDOW:
        return []
    smoothed = savgol_filter(signal, SMOOTHING_WINDOW, polyorder=3)

    # Find peaks (extended arm positions) in the smoothed signal
    # distance = minimum samples between peaks
    # prominence = how much a peak stands out from surrounding data
    peaks, properties = find_peaks(
        smoothed,
        distance=MIN_REP_SAMPLES,
        prominence=0.3  # Adjust if too many/few reps detected
    )

    # Each rep = data from one peak to the next
    reps = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        rep_data = session_df.iloc[start:end].reset_index(drop=True)

        # Only keep reps that are a reasonable length (1-6 seconds)
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
    More features than the window approach because we have the full rep.
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
        features[f"{col}_skew"] = pd.Series(values).skew()    # Asymmetry of the curve
        features[f"{col}_kurtosis"] = pd.Series(values).kurtosis()  # Peakedness

    # Rep duration in seconds — slow or fast reps have different durations
    features["rep_duration_s"] = len(rep_df) / SAMPLE_RATE

    # Total movement energy — sum of squared acceleration
    # Higher energy = more aggressive/jerky movement
    accel_magnitude = np.sqrt(
        rep_df["accel_x_ms2"].values**2 +
        rep_df["accel_y_ms2"].values**2 +
        rep_df["accel_z_ms2"].values**2
    )
    features["accel_energy"] = np.sum(accel_magnitude**2) / len(rep_df)

    # Smoothness — standard deviation of the gyroscope magnitude
    # Jerky movement = high gyro variation
    gyro_magnitude = np.sqrt(
        rep_df["gyro_x_rads"].values**2 +
        rep_df["gyro_y_rads"].values**2 +
        rep_df["gyro_z_rads"].values**2
    )
    features["gyro_smoothness"] = np.std(gyro_magnitude)

    return features


# ============================================================
# STEP 4 — Build the per-rep dataset
# ============================================================

def build_rep_dataset(combined_df):
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

        # Detect individual reps in this session
        reps = detect_reps(session)

        for rep_df in reps:
            features = extract_rep_features(rep_df)
            features["label"] = label
            all_rep_features.append(features)
            total_reps += 1
            if label == "good":
                good_reps += 1
            else:
                bad_reps += 1

        print(f"  {source_file}: found {len(reps)} reps ({label})")

    rep_dataset = pd.DataFrame(all_rep_features)
    print(f"\nTotal reps detected: {total_reps}")
    print(f"  Good reps: {good_reps}")
    print(f"  Bad reps:  {bad_reps}")
    return rep_dataset


# ============================================================
# STEP 5 — Train/test split
# ============================================================

def split_data(rep_dataset):
    """Separate features from labels, split into train/test."""
    X = rep_dataset.drop("label", axis=1)
    y = rep_dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print(f"Training set: {len(X_train)} reps")
    print(f"Test set:     {len(X_test)} reps")
    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 6 — Train the Random Forest
# ============================================================

def train_model(X_train, y_train):
    """Train a Random Forest classifier on per-rep features."""
    model = RandomForestClassifier(
        n_estimators=200,        # 200 trees for more stable predictions
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# STEP 7 — Evaluate
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Test the model on reps it has never seen."""
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\n{'='*50}")
    print(f"  PER-REP ACCURACY: {accuracy * 100:.1f}%")
    print(f"{'='*50}\n")

    print("Detailed report:")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions, labels=["good", "bad"])
    print("Confusion matrix:")
    print(f"                 Predicted Good   Predicted Bad")
    print(f"Actual Good         {cm[0][0]:5d}            {cm[0][1]:5d}")
    print(f"Actual Bad          {cm[1][0]:5d}            {cm[1][1]:5d}")

    return predictions, cm


# ============================================================
# STEP 8 — Visualize
# ============================================================

def plot_results(model, X_train, cm):
    """Show confusion matrix and feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Confusion matrix
    ax1 = axes[0]
    ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Good", "Bad"])
    ax1.set_yticklabels(["Good", "Bad"])
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Per-Rep Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, cm[i][j], ha="center", va="center",
                     color="white" if cm[i][j] > cm.max() / 2 else "black",
                     fontsize=14, fontweight="bold")

    # Top 20 most important features
    ax2 = axes[1]
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top_features = importances.nlargest(20).sort_values()
    ax2.barh(range(len(top_features)), top_features.values, color="steelblue")
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features.index, fontsize=8)
    ax2.set_xlabel("Importance")
    ax2.set_title("Top 20 Most Important Features (Per-Rep)")

    plt.tight_layout()
    plt.savefig("rep_results.png", dpi=120)
    print("\nResults saved to rep_results.png")
    plt.show()


# ============================================================
# STEP 9 — Plot example rep detection from one session
# ============================================================

def plot_example_reps(combined_df):
    """Show rep detection on one session so you can verify it's working."""
    # Grab the first good session
    first_file = combined_df[combined_df["label"] == "good"]["source_file"].unique()[0]
    session = combined_df[combined_df["source_file"] == first_file].reset_index(drop=True)

    signal = session[REP_DETECT_COL].values
    smoothed = savgol_filter(signal, SMOOTHING_WINDOW, polyorder=3)

    peaks, _ = find_peaks(smoothed, distance=MIN_REP_SAMPLES, prominence=0.5)

    time_axis = np.arange(len(signal)) / SAMPLE_RATE

    plt.figure(figsize=(14, 4))
    plt.plot(time_axis, signal, alpha=0.4, label="Raw accel_x")
    plt.plot(time_axis, smoothed, linewidth=2, label="Smoothed")
    plt.plot(time_axis[peaks], smoothed[peaks], "rv", markersize=10, label=f"Rep boundaries ({len(peaks)} found)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration X (m/s²)")
    plt.title(f"Rep Detection — {first_file}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rep_detection_example.png", dpi=120)
    print(f"Rep detection example saved to rep_detection_example.png")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n--- STEP 1: Loading data ---")
    combined = load_all_data()

    print("\n--- STEP 2: Visualizing rep detection ---")
    plot_example_reps(combined)

    print("\n--- STEP 3: Detecting reps and extracting features ---")
    rep_dataset = build_rep_dataset(combined)

    print("\n--- STEP 4: Splitting train/test ---")
    X_train, X_test, y_train, y_test = split_data(rep_dataset)

    print("\n--- STEP 5: Training Random Forest ---")
    model = train_model(X_train, y_train)

    print("\n--- STEP 6: Evaluating ---")
    predictions, cm = evaluate_model(model, X_test, y_test)

    print("\n--- STEP 7: Visualizing results ---")
    plot_results(model, X_train, cm)

    # Save the model and rep detection settings
    joblib.dump(model, "rep_classifier.pkl")
    print("\nModel saved to rep_classifier.pkl")

    return model


if __name__ == "__main__":
    model = main()