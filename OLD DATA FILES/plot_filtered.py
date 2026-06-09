import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("sensor_data_labeled.csv")
df = df[df["glitch_flag"] == 0].reset_index(drop=True)
df["time_s"] = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0

# ── Feature engineering ────────────────────────────────────────────────────────
df["accel_mag"] = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2)
df["gyro_mag"]  = np.sqrt(df["gyro_x"]**2  + df["gyro_y"]**2  + df["gyro_z"]**2)
df["flex_norm"] = (df["flex_voltage"] - df["flex_voltage"].min()) / \
                  (df["flex_voltage"].max() - df["flex_voltage"].min())

# ── High-pass filter ───────────────────────────────────────────────────────────
# Removes slow drift and gravity from the accelerometer
# Keeps only the fast motion components (the actual curl movement)

# Calculate sample rate from timestamps
sample_rate = 1.0 / df["time_s"].diff().median()  # Hz
print(f"Sample rate: {sample_rate:.1f} Hz")

def highpass_filter(data, cutoff_hz, sample_rate, order=4):
    """
    cutoff_hz  — frequencies BELOW this are removed
                 e.g. 0.5 Hz removes drift but keeps curl motion
    order      — filter sharpness (4 is a good default)
    """
    nyquist = sample_rate / 2.0
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)  # filtfilt = zero phase delay (no time shift)

CUTOFF_HZ = 0.5  # tune this — lower = remove less, higher = remove more

df["accel_x_filt"]   = highpass_filter(df["accel_x"],   CUTOFF_HZ, sample_rate)
df["accel_y_filt"]   = highpass_filter(df["accel_y"],   CUTOFF_HZ, sample_rate)
df["accel_z_filt"]   = highpass_filter(df["accel_z"],   CUTOFF_HZ, sample_rate)
df["gyro_mag_filt"]  = highpass_filter(df["gyro_mag"],  CUTOFF_HZ, sample_rate)
df["flex_norm_filt"] = highpass_filter(df["flex_norm"],  0.1, sample_rate)  # lower cutoff for flex

# Filtered accel magnitude
df["accel_mag_filt"] = np.sqrt(
    df["accel_x_filt"]**2 +
    df["accel_y_filt"]**2 +
    df["accel_z_filt"]**2
)

# ── Rep detection (on filtered flex) ──────────────────────────────────────────
peaks_raw, _ = find_peaks(df["flex_norm"],      prominence=0.3, distance=50)
peaks_filt, _ = find_peaks(df["flex_norm_filt"], prominence=0.05, distance=50)

print(f"Reps detected (raw):      {len(peaks_raw)}")
print(f"Reps detected (filtered): {len(peaks_filt)}")

# ── Plotting ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)
fig.suptitle(f"Bicep Curl — High-Pass Filtered (cutoff: {CUTOFF_HZ} Hz)",
             fontsize=15, fontweight="bold")

# --- Plot 1: Flex raw vs filtered ---
axes[0].plot(df["time_s"], df["flex_norm"],      color="lightsteelblue", linewidth=1,
             label="Flex raw", alpha=0.7)
axes[0].plot(df["time_s"], df["flex_norm_filt"], color="royalblue",      linewidth=1.5,
             label="Flex filtered")
axes[0].plot(df["time_s"].iloc[peaks_filt], df["flex_norm_filt"].iloc[peaks_filt],
             "v", color="red", markersize=9, label=f"Reps detected ({len(peaks_filt)})")
axes[0].set_ylabel("Flex (norm)")
axes[0].set_title("Flex Sensor — Raw vs Filtered")
axes[0].legend(loc="upper right")
axes[0].grid(alpha=0.3)

# --- Plot 2: Accel X raw vs filtered ---
axes[1].plot(df["time_s"], df["accel_x"],      color="lightcoral",  linewidth=1,
             label="Accel X raw", alpha=0.7)
axes[1].plot(df["time_s"], df["accel_x_filt"], color="firebrick",   linewidth=1.5,
             label="Accel X filtered")
axes[1].set_ylabel("m/s²")
axes[1].set_title("Accelerometer X — Raw vs Filtered (gravity removed)")
axes[1].legend(loc="upper right")
axes[1].grid(alpha=0.3)

# --- Plot 3: Accel magnitude raw vs filtered ---
axes[2].plot(df["time_s"], df["accel_mag"],      color="moccasin",    linewidth=1,
             label="Accel mag raw", alpha=0.7)
axes[2].plot(df["time_s"], df["accel_mag_filt"], color="darkorange",  linewidth=1.5,
             label="Accel mag filtered")
for t in df["time_s"].iloc[peaks_filt].values:
    axes[2].axvline(x=t, color="red", linestyle="--", alpha=0.4)
axes[2].set_ylabel("m/s²")
axes[2].set_title("Acceleration Magnitude — Raw vs Filtered")
axes[2].legend(loc="upper right")
axes[2].grid(alpha=0.3)

# --- Plot 4: Gyro magnitude raw vs filtered ---
axes[3].plot(df["time_s"], df["gyro_mag"],      color="lightgreen",  linewidth=1,
             label="Gyro mag raw", alpha=0.7)
axes[3].plot(df["time_s"], df["gyro_mag_filt"], color="darkgreen",   linewidth=1.5,
             label="Gyro mag filtered")
for t in df["time_s"].iloc[peaks_filt].values:
    axes[3].axvline(x=t, color="red", linestyle="--", alpha=0.4)
axes[3].set_xlabel("Time (seconds)")
axes[3].set_ylabel("rad/s")
axes[3].set_title("Gyroscope Magnitude — Raw vs Filtered")
axes[3].legend(loc="upper right")
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("curl_filtered.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Rep summary ────────────────────────────────────────────────────────────────
rep_times = df["time_s"].iloc[peaks_filt].values
if len(rep_times) > 1:
    intervals = np.diff(rep_times)
    print(f"\nRep interval stats:")
    print(f"  Average time between reps : {intervals.mean():.2f}s")
    print(f"  Fastest rep               : {intervals.min():.2f}s")
    print(f"  Slowest rep               : {intervals.max():.2f}s")