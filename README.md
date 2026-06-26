# Smart Workout Armband

A wearable fitness device that classifies exercise form in real time using IMU sensor data, BLE communication, and a Python-based ML pipeline with a live Streamlit dashboard.

Built for TECHIN515 at the Global Innovation Exchange (GIX) / University of Washington.

## Team

- **Coleman** — HW/SW & ML Pipeline
- **Kitty** — Systems/Integration/PCB/Enclosure
- **Veronika** — HW/SW Integration/Datasets
- **Daniel** — Enclosure/HW/ Datasets

---

<img width="3024" height="4032" alt="IMG_4101" src="https://github.com/user-attachments/assets/e1d20195-5a63-43d4-b246-106e80d4f8fa" />
<img width="3024" height="4032" alt="IMG_4104" src="https://github.com/user-attachments/assets/3d42f55c-fd01-447a-9ef2-113c57689a84" />
<img width="1285" height="903" alt="image" src="https://github.com/user-attachments/assets/a3c04bde-0ec8-4ff3-adf3-1b7e7ac20428" />



## Overview

The armband straps to the user's wrist and streams 9-axis IMU data over Bluetooth Low Energy (BLE) to a laptop. A Python script detects individual exercise reps from the accelerometer signal, extracts 75 features per rep, and classifies each rep as **good form** or **bad form** using a per-exercise Random Forest model. Results are displayed on a live Streamlit dashboard and fed back to the device's NeoPixel LED — green flash for good form, red flash for bad form.

### Supported Exercises

| Exercise | Rep Detection Axis | Direction |
|---|---|---|
| Bicep Curl | accel_z | valleys |
| Dumbbell Row | accel_y | peaks |
| Lat Raise | accel_z | peaks |
| Shoulder Press | accel_x | valleys |

Each exercise has its own independently trained classifier. The user selects the exercise from the dashboard dropdown, and the system automatically loads the correct model and detection axis.

---

## Hardware

### Components

- **Microcontroller**: Seeed XIAO ESP32-S3
- **IMU**: ICM-20948 9-DoF (accelerometer, gyroscope, magnetometer) via I2C
- **LED**: WS2812B NeoPixel (single pixel, pin D0)
- **Power**: LiPo battery (JST 1.25mm connector) or USB-C
- **Communication**: BLE 4.2 via ESP32-S3 radio

### PCB Versions

Two PCB iterations were built during the project:

**PCB v1** — IMU at I2C address 0x69 (AD0 pulled high). Experienced progressive IMU failures under BLE load due to insufficient power supply decoupling near the IMU. Diagnosed via a custom diagnostic sketch (`ble_imu_diagnostic.ino`) that tracked zero-reading events, recovery attempts, and correlation with BLE connection state. Root cause: BLE radio current spikes (50–150mA) cause voltage droops below the ICM-20948's minimum operating voltage, corrupting the sensor's internal state. Thermal stress from the ESP32-S3 positioned adjacent to the IMU on the PCB contributed to decreasing time-to-failure across power cycles.

**PCB v2** — IMU at I2C address 0x68 (AD0 pulled low). Fresh build to resolve the degradation issues from v1. Recommended hardware fix for future revisions: 10µF tantalum + 100nF ceramic decoupling capacitor pair placed directly at the IMU's VDD pin.

### Pin Mapping

| Function | Pin |
|---|---|
| I2C SDA | GPIO5 |
| I2C SCL | GPIO6 |
| NeoPixel Data | D0 |

### NeoPixel Status Codes

| Color | Meaning |
|---|---|
| Orange | Advertising — waiting for BLE connection |
| Blue | Connected — streaming data, awaiting rep classification |
| Green flash (3x) | Good form rep classified |
| Red flash (3x) | Bad form rep classified |
| Rapid red blink | IMU not found on I2C bus (fatal) |

### Resolved — Battery + BLE (PCB v1)

During development, PCB v1 experienced intermittent IMU zero-readings when running on battery power with BLE active. The failure was progressive (time-to-failure decreased across power cycles) and required a full power-off to recover. USB power provided stable operation indefinitely. The issue was diagnosed using a custom diagnostic sketch (`ble_imu_diagnostic.ino`) that tracked zero-reading events, recovery timing, and BLE connection state correlation. Root cause was attributed to insufficient bulk capacitance on the PCB near the IMU, causing BLE transmit current spikes to droop the I2C supply voltage.

**Resolution**: A new PCB (v2) was built with fresh components, resolving the issue. The firmware retains auto-recovery logic (IMU re-initialization on zero detection) as a safety net.

---

## Software Architecture

```
workout-armband/
├── data/
│   ├── Bicep_curl/
│   │   ├── good/          ← good form CSVs
│   │   └── bad/           ← bad form CSVs
│   ├── Dumbbell Row/
│   │   ├── good/
│   │   └── bad/
│   ├── Lat_raise/
│   │   ├── good/
│   │   └── bad/
│   └── Shoulder_press/
│       ├── good/
│       └── bad/
├── models/
│   ├── bicep_curl_classifier.pkl
│   ├── dumbbell_row_classifier.pkl
│   ├── lat_raise_classifier.pkl
│   └── shoulder_press_classifier.pkl
├── train_all_exercises.py      ← trains all 4 models
├── live_rep_predict_ble.py     ← BLE receiver + live classification
├── dashboard.py                ← Streamlit live dashboard
├── capture_ble.py              ← BLE data capture to CSV
├── diagnose_axes.py            ← finds optimal axis per exercise
└── session_log.json            ← shared state between predict + dashboard
```

### Arduino Firmware

`workout_armband_final_R_G.ino` — Runs on the XIAO ESP32-S3. Streams 11-value CSV lines over BLE at 20Hz:

```
elapsed_ms, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, temp
```

Receives classification feedback ("good"/"bad") on a second BLE characteristic and flashes the NeoPixel accordingly. Includes zero-detection with auto-recovery.

Two BLE characteristics on service `12345678-1234-1234-1234-123456789abc`:
- `abcdefab-cdef-abcd-efab-cdefabcdefab` — sensor data (NOTIFY, Arduino → laptop)
- `abcdefab-cdef-abcd-efab-cdefabcdef00` — feedback (WRITE, laptop → Arduino)

### Data Capture

`capture_ble.py` — Connects to the WorkoutArmband BLE device, receives sensor data, and writes CSV files. Adds a human-readable timestamp as the first column, resulting in 12-column CSVs (or 14 for legacy files that included flex sensor data).

### Training Pipeline

`train_all_exercises.py` — Trains a separate Random Forest classifier for each exercise:

1. **Load**: Reads all CSVs from each exercise's `good/` and `bad/` folders. Handles variable column counts (11, 12, 14) by reading with headers and dropping legacy flex sensor columns.
2. **Trim**: Removes the first 5 seconds and last 2 seconds of each session to exclude noisy setup/teardown.
3. **Rep detection**: Segments individual reps using `scipy.signal.find_peaks` on a per-exercise accelerometer axis with Savitzky-Golay smoothing (window=11, prominence=0.3, min distance=1.0s). Valid reps are 1–6 seconds long.
4. **Feature extraction**: 75 features per rep — 8 statistical measures (mean, std, min, max, range, median, skew, kurtosis) across 9 sensor channels, plus rep duration, acceleration energy, and gyroscope smoothness.
5. **Training**: Random Forest with 200 estimators, 80/20 train/test split with stratification.
6. **Output**: Saves `.pkl` model files to `models/` and prints accuracy, confusion matrix, and classification report.

### Live Prediction

`live_rep_predict_ble.py` — Connects to the armband over BLE, buffers incoming sensor data, detects completed reps using the same per-exercise axis configuration as training, extracts features, and classifies with the corresponding model. Writes results to `session_log.json` for the dashboard, and sends feedback back to the device NeoPixel via BLE write.

Loads all four models at startup for instant exercise switching via the dashboard dropdown. Exercise changes are detected by polling `session_log.json` once per second.

### Dashboard

`dashboard.py` — Streamlit app that reads `session_log.json` and displays:
- Exercise selector dropdown (sidebar)
- Summary metrics: total reps, good reps, bad reps, form score %
- Last rep verdict with confidence and duration
- Full rep history table with color-coded results
- Rep duration bar chart

Auto-refreshes every second.

---

## ML Approach and Key Decisions

### Per-Rep Classification

The system evaluates each complete rep as a unit rather than using windowed/second-by-second classification. This is the correct granularity for exercise form — a single time slice within a rep is not meaningful for judging form quality.

### Per-Exercise Models

Each exercise has its own independent classifier rather than a single multi-exercise model. This avoids confusion between exercises that have similar motion signatures and allows each model to specialize.

### Per-Exercise Axis Configuration

Different exercises produce their strongest periodic signal on different accelerometer axes and directions. The optimal axis per exercise was determined empirically using `diagnose_axes.py`, which tests all 3 axes in both directions on training data and reports which finds the most valid reps. Training and live prediction use identical axis configurations to ensure rep boundaries match.

### Cross-Exercise Bad Form Data

Bad form training data intentionally includes recordings of other exercises. For example, good lat raises are used as bad shoulder press data, and vice versa. This broadens the "bad" class to mean "anything that isn't a correct rep of this exercise" rather than a narrow set of specific mistakes, improving robustness.

### Data Leakage Awareness

Test accuracy (97–99%+) is partially inflated because reps from the same recording session can appear in both train and test splits. True validation is live performance, which is why the live BLE pipeline was prioritized for testing.

### Device Orientation Sensitivity

The classifier is sensitive to IMU orientation. Training data must be recorded with the device in the same physical orientation as live use. The bicep curl dataset was re-recorded after a device orientation change was identified as the cause of poor live classification accuracy.

---

## Setup and Usage

### Prerequisites

```
pip install bleak scikit-learn pandas numpy scipy joblib streamlit
```

Arduino IDE with ESP32 board support and libraries: Adafruit ICM20X, Adafruit NeoPixel, ESP32 BLE Arduino.

### 1. Flash the Arduino Sketch

Open `workout_armband_final_R_G.ino` in Arduino IDE, select XIAO_ESP32S3, and upload. Verify the NeoPixel shows orange (advertising).

### 2. Record Training Data

```bash
python capture_ble.py
```

Record multiple sessions of good form and bad form for each exercise. Place CSVs in the appropriate `data/{Exercise}/good/` or `data/{Exercise}/bad/` folders.

Aim for 15–20+ sessions per class per exercise for robust training.

### 3. Determine Optimal Axes

```bash
python diagnose_axes.py
```

Reports which accelerometer axis and direction finds the best reps for each exercise. Update `EXERCISE_AXES` in both `train_all_exercises.py` and `live_rep_predict_ble.py` if axes change.

### 4. Train Models

```bash
python train_all_exercises.py
```

Trains all four models and saves them to `models/`. Prints accuracy and confusion matrices.

### 5. Run Live System

Terminal 1:
```bash
python live_rep_predict_ble.py
```

Terminal 2:
```bash
streamlit run dashboard.py
```

Select the exercise from the dashboard dropdown. Perform reps — the terminal shows classifications, the dashboard updates in real time, and the NeoPixel flashes green or red on the device.

---

## Diagnostic and Test Sketches

| Sketch | Purpose |
|---|---|
| `i2c_scanner.ino` | Scans I2C bus for connected devices. Verifies IMU address. |
| `imu_test.ino` | Reads IMU without BLE — isolates hardware vs software issues. |
| `neopixel_test.ino` | Cycles NeoPixel through colors — verifies LED functionality. |
| `imu_neopixel_test.ino` | Combined IMU + NeoPixel test without BLE. |
| `ble_imu_diagnostic.ino` | Full BLE + IMU diagnostic with zero tracking, recovery attempts, and timing analysis. Used to diagnose the PCB v1 power issue. |

---

## Current Status

- **ML pipeline**: All four exercise classifiers trained and producing accurate live predictions.
- **BLE feedback loop**: Complete — predictions flow from laptop back to device NeoPixel.
- **Dashboard**: Functional with exercise dropdown, live metrics, and rep history.
- **Hardware**: PCB v2 built with corrected I2C address (0x68). Battery stability under investigation; USB operation is fully stable.
- **Remaining**: Final bicep curl and dumbbell row training data collection with consistent device orientation, then retrain.
