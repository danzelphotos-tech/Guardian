import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("sensor_data_labeled.csv")

# Drop glitch rows
df = df[df["glitch_flag"] == 0].reset_index(drop=True)

# Convert timestamp to seconds for cleaner x-axis
df["time_s"] = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0

# ── Feature engineering ────────────────────────────────────────────────────────
# Accelerometer magnitude — captures overall movement intensity
df["accel_mag"] = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2)

# Gyroscope magnitude — captures rotation speed of the curl
df["gyro_mag"] = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2)

# Normalize flex voltage to 0–1 for easier comparison
df["flex_norm"] = (df["flex_voltage"] - df["flex_voltage"].min()) / \
                  (df["flex_voltage"].max() - df["flex_voltage"].min())

# ── Rep detection ──────────────────────────────────────────────────────────────
# A bicep curl rep = flex sensor peaks (arm fully curled)
# Tune `prominence` and `distance` based on your data
peaks, properties = find_peaks(
    df["flex_norm"],
    prominence=0.3,   # minimum peak height above surroundings
    distance=50       # minimum samples between peaks (~1 rep)
)

rep_times = df["time_s"].iloc[peaks].values
print(f"Detected {len(peaks)} curl reps")
print(f"Rep times (s): {np.round(rep_times, 2)}")

# ── Plotting ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Bicep Curl Detection", fontsize=16, fontweight="bold")

# --- Plot 1: Flex sensor (primary curl indicator) ---
axes[0].plot(df["time_s"], df["flex_norm"], color="royalblue", linewidth=1.5, label="Flex (normalized)")
axes[0].plot(df["time_s"].iloc[peaks], df["flex_norm"].iloc[peaks],
             "v", color="red", markersize=10, label=f"Detected reps ({len(peaks)})")
axes[0].set_ylabel("Flex (normalized)")
axes[0].set_title("Flex Sensor — Arm Curl Position")
axes[0].legend(loc="upper right")
axes[0].grid(alpha=0.3)

# --- Plot 2: Accelerometer axes ---
axes[1].plot(df["time_s"], df["accel_x"], label="X", alpha=0.8)
axes[1].plot(df["time_s"], df["accel_y"], label="Y", alpha=0.8)
axes[1].plot(df["time_s"], df["accel_z"], label="Z", alpha=0.8)
axes[1].set_ylabel("m/s²")
axes[1].set_title("Accelerometer (X, Y, Z)")
axes[1].legend(loc="upper right")
axes[1].grid(alpha=0.3)

# --- Plot 3: Accel magnitude + rep markers ---
axes[2].plot(df["time_s"], df["accel_mag"], color="darkorange", linewidth=1.5, label="Accel magnitude")
for t in rep_times:
    axes[2].axvline(x=t, color="red", linestyle="--", alpha=0.5)
axes[2].set_ylabel("m/s²")
axes[2].set_title("Acceleration Magnitude (with rep markers)")
axes[2].legend(loc="upper right")
axes[2].grid(alpha=0.3)

# --- Plot 4: Gyroscope magnitude ---
axes[3].plot(df["time_s"], df["gyro_mag"], color="green", linewidth=1.5, label="Gyro magnitude")
for t in rep_times:
    axes[3].axvline(x=t, color="red", linestyle="--", alpha=0.5)
axes[3].set_xlabel("Time (seconds)")
axes[3].set_ylabel("rad/s")
axes[3].set_title("Gyroscope Magnitude (rotation speed)")
axes[3].legend(loc="upper right")
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("curl_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Rep summary stats ──────────────────────────────────────────────────────────
if len(peaks) > 1:
    intervals = np.diff(rep_times)
    print(f"\nRep interval stats:")
    print(f"  Average time between reps: {intervals.mean():.2f}s")
    print(f"  Fastest rep: {intervals.min():.2f}s")
    print(f"  Slowest rep: {intervals.max():.2f}s")