import serial
import time
from datetime import datetime

PORT = "COM5"
BAUD = 115200
DURATION_SEC = 70

# Name your file based on what you're recording
FILENAME = "v2_bicep_curl_VeronikaGood_form2.csv"

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # Wait for ESP32 to reset

print(f"Recording to {FILENAME} for {DURATION_SEC} seconds...")
print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
print("Start your reps and your video NOW!")

# Write a clear, labeled header
header = (
    "timestamp,"
    "elapsed_ms,"
    "accel_x_ms2,"
    "accel_y_ms2,"
    "accel_z_ms2,"
    "gyro_x_rads,"
    "gyro_y_rads,"
    "gyro_z_rads,"
    "mag_x_uT,"
    "mag_y_uT,"
    "mag_z_uT,"
    "temp_c,"
)

start = time.time()
lines_written = 0
header_skipped = False

with open(FILENAME, "w") as f:
    f.write(header + "\n")

    while time.time() - start < DURATION_SEC:
        line = ser.readline().decode("utf-8").strip()
        if not line or line.startswith("ERROR"):
            continue

        # Skip the header line that the Arduino prints
        if not header_skipped and "timestamp_ms" in line:
            header_skipped = True
            continue

        # Add real timestamp to the front of each row
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # e.g. 14:32:05.123
        f.write(now + "," + line + "\n")
        lines_written += 1

        if lines_written % 100 == 0:
            elapsed = int(time.time() - start)
            print(f"  {elapsed}s — {lines_written} rows captured")

ser.close()
end_time = datetime.now().strftime("%H:%M:%S")
print(f"\nDone! Ended at: {end_time}")
print(f"Saved {lines_written} rows to {FILENAME}")
print(f"\nSync tip: match the 'timestamp' column to your video timecode")