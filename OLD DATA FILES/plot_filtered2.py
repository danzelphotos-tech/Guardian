import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("sensor_data_labeled.csv")
df = df[df["glitch_flag"] == 0].reset_index(drop=True)
df["time_s"] = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0

df["flex_norm"] = (df["flex_voltage"] - df["flex_voltage"].min()) / \
                  (df["flex_voltage"].max() - df["flex_voltage"].min())

df["accel_mag"] = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2)
df["gyro_mag"]  = np.sqrt(df["gyro_x"]**2  + df["gyro_y"]**2  + df["gyro_z"]**2)

# ── Filter setup ───────────────────────────────────────────────────────────────
sample_rate = 1.0 / df["time_s"].diff().median()
print(f"Sample rate: {sample_rate:.1f} Hz")

CUTOFF_HZ = 0.5

def highpass_filter(data, cutoff_hz, sample_rate, order=4):
    nyquist = sample_rate / 2.0
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)

# Filter every signal
df["accel_x_filt"]   = highpass_filter(df["accel_x"],   CUTOFF_HZ, sample_rate)
df["accel_y_filt"]   = highpass_filter(df["accel_y"],   CUTOFF_HZ, sample_rate)
df["accel_z_filt"]   = highpass_filter(df["accel_z"],   CUTOFF_HZ, sample_rate)
df["gyro_x_filt"]    = highpass_filter(df["gyro_x"],    CUTOFF_HZ, sample_rate)
df["gyro_y_filt"]    = highpass_filter(df["gyro_y"],    CUTOFF_HZ, sample_rate)
df["gyro_z_filt"]    = highpass_filter(df["gyro_z"],    CUTOFF_HZ, sample_rate)
df["accel_mag_filt"] = highpass_filter(df["accel_mag"], CUTOFF_HZ, sample_rate)
df["gyro_mag_filt"]  = highpass_filter(df["gyro_mag"],  CUTOFF_HZ, sample_rate)
df["flex_filt"]      = highpass_filter(df["flex_norm"],  0.1,       sample_rate)

# ── Rep detection ──────────────────────────────────────────────────────────────
peaks, _ = find_peaks(df["flex_norm"], prominence=0.3, distance=50)
rep_times = df["time_s"].iloc[peaks].values
print(f"Reps detected: {len(peaks)}")

# ── Helper — draw rep lines on any axis ───────────────────────────────────────
def draw_reps(ax):
    for t in rep_times:
        ax.axvline(x=t, color="red", linestyle="--", alpha=0.4, linewidth=1)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Flex Sensor
# ══════════════════════════════════════════════════════════════════════════════
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig1.suptitle("Flex Sensor — Raw vs High-Pass Filtered", fontsize=14, fontweight="bold")

ax1.plot(df["time_s"], df["flex_norm"], color="lightsteelblue", linewidth=1, label="Raw")
ax1.plot(df["time_s"].iloc[peaks], df["flex_norm"].iloc[peaks],
         "v", color="red", markersize=9, label=f"Reps ({len(peaks)})")
ax1.set_ylabel("Flex (normalized)")
ax1.set_title("Raw")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)
draw_reps(ax1)

ax2.plot(df["time_s"], df["flex_filt"], color="royalblue", linewidth=1.5, label="Filtered")
ax2.set_ylabel("Flex (filtered)")
ax2.set_xlabel("Time (s)")
ax2.set_title("High-Pass Filtered")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)
draw_reps(ax2)

plt.tight_layout()
plt.savefig("fig1_flex.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Accelerometer X
# ══════════════════════════════════════════════════════════════════════════════
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig2.suptitle("Accelerometer X — Raw vs High-Pass Filtered", fontsize=14, fontweight="bold")

ax1.plot(df["time_s"], df["accel_x"], color="lightsalmon", linewidth=1, label="Raw")
ax1.set_ylabel("m/s²")
ax1.set_title("Raw")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)
draw_reps(ax1)

ax2.plot(df["time_s"], df["accel_x_filt"], color="firebrick", linewidth=1.5, label="Filtered")
ax2.set_ylabel("m/s²")
ax2.set_xlabel("Time (s)")
ax2.set_title("High-Pass Filtered (gravity removed)")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)
draw_reps(ax2)

plt.tight_layout()
plt.savefig("fig2_accel_x.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Accelerometer Y
# ══════════════════════════════════════════════════════════════════════════════
fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig3.suptitle("Accelerometer Y — Raw vs High-Pass Filtered", fontsize=14, fontweight="bold")

ax1.plot(df["time_s"], df["accel_y"], color="lightsalmon", linewidth=1, label="Raw")
ax1.set_ylabel("m/s²")
ax1.set_title("Raw")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)
draw_reps(ax1)

ax2.plot(df["time_s"], df["accel_y_filt"], color="firebrick", linewidth=1.5, label="Filtered")
ax2.set_ylabel("m/s²")
ax2.set_xlabel("Time (s)")
ax2.set_title("High-Pass Filtered (gravity removed)")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)
draw_reps(ax2)

plt.tight_layout()
plt.savefig("fig3_accel_y.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Accelerometer Z
# ══════════════════════════════════════════════════════════════════════════════
fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig4.suptitle("Accelerometer Z — Raw vs High-Pass Filtered", fontsize=14, fontweight="bold")

ax1.plot(df["time_s"], df["accel_z"], color="lightsalmon", linewidth=1, label="Raw")
ax1.set_ylabel("m/s²")
ax1.set_title("Raw")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)
draw_reps(ax1)

ax2.plot(df["time_s"], df["accel_z_filt"], color="firebrick", linewidth=1.5, label="Filtered")
ax2.set_ylabel("m/s²")
ax2.set_xlabel("Time (s)")
ax2.set_title("High-Pass Filtered (gravity removed)")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)
draw_reps(ax2)

plt.tight_layout()
plt.savefig("fig4_accel_z.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Accelerometer Magnitude
# ══════════════════════════════════════════════════════════════════════════════
fig5, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig5.suptitle("Accelerometer Magnitude — Raw vs High-Pass Filtered", fontsize=14, fontweight="bold")

ax1.plot(df["time_s"], df["accel_mag"], color="moccasin", linewidth=1, label="Raw")
ax1.set_ylabel("m/s²")
ax1.set_title("Raw")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)
draw_reps(ax1)

ax2.plot(df["time_s"], df["accel_mag_filt"], color="darkorange", linewidth=1.5, label="Filtered")
ax2.set_ylabel("m/s²")
ax2.set_xlabel("Time (s)")
ax2.set_title("High-Pass Filtered")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)
draw_reps(ax2)

plt.tight_layout()
plt.savefig("fig5_accel_mag.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Gyroscope X
# ══════════════════════════════════════════════════════════════════════════════
fig6, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig6.suptitle("Gyroscope X — Raw vs High-Pass Filtered", fontsize=14, fontweight="bold")

ax1.plot(df["time_s"], df["gyro_x"], color="lightgreen", linewidth=1, label="Raw")
ax1.set_ylabel("rad/s")
ax1.set_title("Raw")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)
draw_reps(ax1)

ax2.plot(df["time_s"], df["gyro_x_filt"], color="darkgreen", linewidth=1.5, label="Filtered")
ax2.set_ylabel("rad/s")
ax2.set_xlabel("Time (s)")
ax2.set_title("High-Pass Filtered")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)
draw_reps(ax2)

plt.tight_layout()
plt.savefig("fig6_gyro_x.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Gyroscope Y
# ══════════════════════════════════════════════════════════════════════════════
fig7, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig7.suptitle("Gyroscope Y — Raw vs High-Pass Filtered", fontsize=14, fontweight="bold")

ax1.plot(df["time_s"], df["gyro_y"], color="lightgreen", linewidth=1, label="Raw")
ax1.set_ylabel("rad/s")
ax1.set_title("Raw")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)
draw_reps(ax1)

ax2.plot(df["time_s"], df["gyro_y_filt"], color="darkgreen", linewidth=1.5, label="Filtered")
ax2.set_ylabel("rad/s")
ax2.set_xlabel("Time (s)")
ax2.set_title("High-Pass Filtered")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)
draw_reps(ax2)

plt.tight_layout()
plt.savefig("fig7_gyro_y.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Gyroscope Z
# ══════════════════════════════════════════════════════════════════════════════
fig8, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig8.suptitle("Gyroscope Z — Raw vs High-Pass Filtered", fontsize=14, fontweight="bold")

ax1.plot(df["time_s"], df["gyro_z"], color="lightgreen", linewidth=1, label="Raw")
ax1.set_ylabel("rad/s")
ax1.set_title("Raw")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)
draw_reps(ax1)

ax2.plot(df["time_s"], df["gyro_z_filt"], color="darkgreen", linewidth=1.5, label="Filtered")
ax2.set_ylabel("rad/s")
ax2.set_xlabel("Time (s)")
ax2.set_title("High-Pass Filtered")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)
draw_reps(ax2)

plt.tight_layout()
plt.savefig("fig8_gyro_z.png", dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Gyroscope Magnitude
# ══════════════════════════════════════════════════════════════════════════════
fig9, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig9.suptitle("Gyroscope Magnitude — Raw vs High-Pass Filtered", fontsize=14, fontweight="bold")

ax1.plot(df["time_s"], df["gyro_mag"], color="lightgreen", linewidth=1, label="Raw")
ax1.set_ylabel("rad/s")
ax1.set_title("Raw")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)
draw_reps(ax1)

ax2.plot(df["time_s"], df["gyro_mag_filt"], color="darkgreen", linewidth=1.5, label="Filtered")
ax2.set_ylabel("rad/s")
ax2.set_xlabel("Time (s)")
ax2.set_title("High-Pass Filtered")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)
draw_reps(ax2)

plt.tight_layout()
plt.savefig("fig9_gyro_mag.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\nAll figures saved:")
for i, name in enumerate([
    "flex", "accel_x", "accel_y", "accel_z", "accel_mag",
    "gyro_x", "gyro_y", "gyro_z", "gyro_mag"
], start=1):
    print(f"  fig{i}_{name}.png")