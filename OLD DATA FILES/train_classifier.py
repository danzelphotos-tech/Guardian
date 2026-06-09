"""
Workout Armband — Bicep Curl Form Classifier
============================================
Trains a Random Forest classifier to distinguish good vs bad form
from IMU (accelerometer + gyroscope + magnetometer) data.

Folder structure expected:
  workout-armband/
    data/
      good/   <- good form CSV files
      bad/    <- bad form CSV files
    train_classifier.py  (this script)
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION — adjust these as needed
# ============================================================

DATA_DIR = "data"            # Folder containing good/ and bad/ subfolders
WINDOW_SIZE = 40             # 40 rows = 2 seconds at 20Hz sampling
WINDOW_STEP = 20             # Slide window by 1 second (50% overlap)
TEST_SIZE = 0.2              # 20% of data held out for testing
RANDOM_SEED = 42             # For reproducibility — same split every run

# Trim noisy data at start and end of each session
SAMPLE_RATE = 20             # 20Hz — matches your Arduino sketch (50ms delay)
TRIM_START_SECONDS = 5       # Skip first 5 seconds (getting into position)
TRIM_END_SECONDS = 2         # Skip last 2 seconds (winding down)

# Sensor columns we'll use as features
# Skipping timestamp_ms and temp_c — they don't help with form detection
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

    print(f"Trimming first {TRIM_START_SECONDS}s ({trim_start_rows} rows) and "
          f"last {TRIM_END_SECONDS}s ({trim_end_rows} rows) of each session")

    for label in ["good", "bad"]:
        folder = os.path.join(DATA_DIR, label)
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        print(f"Found {len(csv_files)} files in {folder}/")

        for filepath in csv_files:
            df = pd.read_csv(filepath)
            original_rows = len(df)

            # Trim noisy start and end
            if len(df) > (trim_start_rows + trim_end_rows + WINDOW_SIZE):
                df = df.iloc[trim_start_rows:-trim_end_rows].reset_index(drop=True)
                df = df.drop(columns=["flex_raw", "flex_voltage_v"], errors="ignore")
            else:
                print(f"  WARNING: {os.path.basename(filepath)} is too short to trim "
                      f"({original_rows} rows) — skipping trim for this file")

            df["label"] = label
            df["source_file"] = os.path.basename(filepath)
            all_dataframes.append(df)

    combined = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total rows after trimming: {len(combined):,}")
    return combined


# ============================================================
# STEP 2 — Feature extraction from sliding windows
# ============================================================

def extract_features_from_window(window):
    """
    Take a small chunk of sensor data (a window) and summarize it
    into one row of features the ML model can learn from.
    """
    features = {}
    for col in SENSOR_COLS:
        values = window[col].values
        features[f"{col}_mean"] = np.mean(values)
        features[f"{col}_std"] = np.std(values)
        features[f"{col}_min"] = np.min(values)
        features[f"{col}_max"] = np.max(values)
        features[f"{col}_range"] = np.max(values) - np.min(values)
    return features


def build_feature_dataset(combined_df):
    """
    Slide a window through each session and convert raw sensor data
    into feature rows. Each window becomes one training example.
    """
    feature_rows = []

    # Process each session file separately so windows don't span across files
    for source_file, session in combined_df.groupby("source_file"):
        session = session.reset_index(drop=True)
        label = session["label"].iloc[0]

        # Slide window across the session
        for start in range(0, len(session) - WINDOW_SIZE, WINDOW_STEP):
            window = session.iloc[start:start + WINDOW_SIZE]
            features = extract_features_from_window(window)
            features["label"] = label
            feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)
    print(f"Created {len(feature_df):,} feature windows")
    print(f"  Good windows: {(feature_df['label'] == 'good').sum():,}")
    print(f"  Bad windows:  {(feature_df['label'] == 'bad').sum():,}")
    return feature_df


# ============================================================
# STEP 3 — Train/test split
# ============================================================

def split_data(feature_df):
    """Separate features (X) from labels (y), then split into train/test."""
    X = feature_df.drop("label", axis=1)
    y = feature_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y  # Keeps good/bad ratio balanced in both sets
    )

    print(f"Training set: {len(X_train):,} windows")
    print(f"Test set:     {len(X_test):,} windows")
    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 4 — Train the Random Forest
# ============================================================

def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,        # 100 decision trees voting
        random_state=RANDOM_SEED,
        n_jobs=-1                # Use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# STEP 5 — Evaluate on the test set
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Test the model on data it has never seen."""
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\n{'='*50}")
    print(f"  ACCURACY: {accuracy * 100:.1f}%")
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
# STEP 6 — Visualize results
# ============================================================

def plot_results(model, X_train, cm):
    """Show feature importance and confusion matrix as charts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrix as heatmap
    ax1 = axes[0]
    ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Good", "Bad"])
    ax1.set_yticklabels(["Good", "Bad"])
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, cm[i][j], ha="center", va="center",
                     color="white" if cm[i][j] > cm.max() / 2 else "black",
                     fontsize=14, fontweight="bold")

    # Top 15 most important features
    ax2 = axes[1]
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top_features = importances.nlargest(15).sort_values()
    ax2.barh(range(len(top_features)), top_features.values, color="steelblue")
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features.index)
    ax2.set_xlabel("Importance")
    ax2.set_title("Top 15 Most Important Features")

    plt.tight_layout()
    plt.savefig("results.png", dpi=120)
    print("\nResults chart saved to results.png")
    plt.show()


# ============================================================
# MAIN — run everything in order
# ============================================================

def main():
    print("\n--- STEP 1: Loading data ---")
    combined = load_all_data()

    print("\n--- STEP 2: Extracting features ---")
    feature_df = build_feature_dataset(combined)

    print("\n--- STEP 3: Splitting train/test ---")
    X_train, X_test, y_train, y_test = split_data(feature_df)

    print("\n--- STEP 4: Training Random Forest ---")
    model = train_model(X_train, y_train)

    print("\n--- STEP 5: Evaluating ---")
    predictions, cm = evaluate_model(model, X_test, y_test)

    print("\n--- STEP 6: Visualizing ---")
    plot_results(model, X_train, cm)


if __name__ == "__main__":
    main()
