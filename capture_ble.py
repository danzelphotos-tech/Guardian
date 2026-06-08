"""
Workout Armband — BLE Data Capture
====================================
Wireless version of capture.py.
Records IMU data over BLE instead of USB serial.

Requirements:
  pip install bleak

Usage:
  python capture_ble.py
"""

import asyncio
from datetime import datetime
from bleak import BleakScanner, BleakClient

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE_NAME         = "WorkoutArmband"
CHARACTERISTIC_UUID = "abcdefab-cdef-abcd-efab-cdefabcdefab"
DURATION_SEC        = 180
FILENAME            = "v3_Bicepcurl_Colegood27.csv"

# CSV header — matches training script column names
HEADER = (
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
    "temp_c"
)


# ============================================================
# MAIN
# ============================================================

async def main():
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

    # Shared state
    lines_written = 0
    start_time = None
    running = True

    async with BleakClient(device) as client:
        print(f"Connected!\n")
        print(f"Recording to {FILENAME} for {DURATION_SEC} seconds...")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
        print("Start your reps NOW!\n")

        with open(FILENAME, "w") as f:
            f.write(HEADER + "\n")

            import time
            start_time = time.time()

            def handle_notification(sender, data):
                nonlocal lines_written, running

                # Stop if duration exceeded
                if time.time() - start_time >= DURATION_SEC:
                    running = False
                    return

                line = data.decode("utf-8").strip()
                if not line:
                    return

                # Add real timestamp to front of each row
                now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                f.write(now + "," + line + "\n")
                lines_written += 1

                # Progress update every 100 lines
                if lines_written % 100 == 0:
                    elapsed = int(time.time() - start_time)
                    print(f"  {elapsed}s — {lines_written} rows captured")

            await client.start_notify(CHARACTERISTIC_UUID, handle_notification)

            # Wait for duration or Ctrl+C
            try:
                while running:
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopped early by user.")

            await client.stop_notify(CHARACTERISTIC_UUID)

    end_time = datetime.now().strftime("%H:%M:%S")
    print(f"\nDone! Ended at: {end_time}")
    print(f"Saved {lines_written} rows to {FILENAME}")


if __name__ == "__main__":
    asyncio.run(main())
