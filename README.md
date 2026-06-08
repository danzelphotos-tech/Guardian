# Smart Workout Armband

TECHIN515 Group Final Project — A wearable armband that detects bicep curl form in real time using an IMU sensor and a machine learning classifier. Each completed rep is classified as good or bad form, displayed on a live Streamlit dashboard.

**Team:** January (HW/SW + ML), Veronika (HW/SW), Coleman (ML), Daniel (Cloud/Dashboard), Kitty (Systems/Integration)

---

## Hardware

| Component | Details |
|-----------|---------|
| Microcontroller | Seeed XIAO ESP32-S3 |
| IMU | Adafruit ICM20948 9-DoF (accelerometer, gyroscope, magnetometer) |
| Feedback | Haptic motor (in progress) |
| Sensor placement | IMU worn on wrist |

### Wiring — ICM20948 to XIAO ESP32-S3

| IMU Pin | XIAO Pin |
|---------|----------|
| VIN | 3V3 |
| GND | GND |
| SDA | D4 (GPIO5) |
| SCL | D5 (GPIO6) |

---

## Repository Structure

```
workout-armband/
├── data/
│   ├── good/                        # Good form CSV sessions
│   └── bad/                         # Bad form CSV sessions
├── arduino_sketch/
│   └── imu_stream.ino               # Arduino firmware
├── capture.py                       # Records sensor data to CSV
├── train_rep_classifier.py          # Trains the ML model
├── live_rep_predict_dashboard.py    # Live prediction backend
├── dashboard.py                     # Streamlit dashboard
├── rep_classifier.pkl               # Trained model (generated)
├── session_log.json                 # Live session data (generated)
└── README.md
```

---

## Installation

### 1. Arduino IDE setup

1. Download [Arduino IDE](https://www.arduino.cc/en/software)
2. Go to **File → Preferences** and add to Additional Board Manager URLs:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Go to **Tools → Board → Board Manager**, search `esp32`, install
4. Select board: **XIAO_ESP32S3**
5. Install libraries via **Sketch → Include Library → Manage Libraries**:
   - `Adafruit ICM20X`
   - `Adafruit BusIO`
   - `Adafruit Unified Sensor`

### 2. Python setup

```bash
pip install pyserial pandas numpy scipy scikit-learn matplotlib joblib streamlit
```

---

## Step 1 — Upload Arduino firmware

Open `arduino_sketch/imu_stream.ino` in Arduino IDE and upload to the XIAO ESP32-S3. The sketch streams CSV-formatted IMU data over serial at 20Hz (50ms per sample).

To verify it is working, open **Serial Monitor** at **115200 baud**. You should see:

```
timestamp_ms,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,temp_c
85098,0.611,4.223,9.041,0.119,-0.039,0.066,-6.75,27.90,7.05,30.5
...
```

The sketch stays on the device permanently — you only need to re-upload if you change the code.

---

## Step 2 — Record training data

> Close Serial Monitor before running any Python script — only one program can use the COM port at a time.

Open `capture.py` and set the filename and port:

```python
PORT = "COM13"           # Change to your port (check Arduino IDE → Tools → Port)
FILENAME = "v2_bicep_curl_ColeGood_form1.csv"
DURATION_SEC = 180       # 3 minutes per session
```

Then run:

```bash
cd C:\Users\YourName\Documents\workout-armband
python capture.py
```

When prompted, start performing reps. The script saves a CSV to your folder automatically.

### Naming convention

Use descriptive filenames so sessions are easy to sort:

- `v2_bicep_curl_ColeGood_form1.csv` — good form session
- `v2_bicep_curl_ColeBad_form1.csv` — bad form session

### Folder setup

After recording, organize your CSVs:

```
data/
├── good/    ← all good form CSVs
└── bad/     ← all bad form CSVs (sloppy curls, shoulder raises, fast reps, etc.)
```

Aim for a roughly balanced dataset — currently 26 good / 29 bad sessions.

---

## Step 3 — Train the model

```bash
python train_rep_classifier.py
```

This will:

1. Load and trim all CSVs (skips first 5s and last 2s of each session)
2. Show a rep detection chart — verify the red markers are landing correctly on peaks, then close the window to continue
3. Detect individual reps using accelerometer peak detection
4. Extract 75 features per rep
5. Train a Random Forest classifier (200 trees)
6. Print accuracy and confusion matrix
7. Save `rep_classifier.pkl` and `rep_results.png`

Expected output:

```
Found 26 files in data/good/
Found 29 files in data/bad/
Total rows after trimming: 135,715

Per-rep accuracy: 99.8%

Confusion matrix:
                 Predicted Good   Predicted Bad
Actual Good           241                1
Actual Bad              0              243

Model saved to rep_classifier.pkl
```

### Key config values (top of script)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `TRIM_START_SECONDS` | 5 | Skips setup noise at start of session |
| `TRIM_END_SECONDS` | 2 | Skips wind-down at end |
| `SAMPLE_RATE` | 20 | Must match Arduino sketch delay (50ms) |
| `MIN_REP_TIME` | 1.0s | Minimum time between detected rep peaks |
| `prominence` | 0.3 | Peak detection sensitivity — lower = more sensitive |

---

## Step 4 — Run live prediction and dashboard

You need two terminals open at the same time.

### Terminal 1 — prediction backend

```bash
cd C:\Users\YourName\Documents\workout-armband
python live_rep_predict_dashboard.py
```

This reads live serial data, detects completed reps, classifies each one, and writes results to `session_log.json`.

### Terminal 2 — Streamlit dashboard

```bash
cd C:\Users\YourName\Documents\workout-armband
streamlit run dashboard.py
```

Streamlit opens a browser tab automatically at `http://localhost:8501`.

### Terminal 1 output

```
Rep # 1  ✓ GOOD FORM  (94% confidence, 2.5s)
Rep # 2  ✓ GOOD FORM  (97% confidence, 2.3s)
Rep # 3  ✗ BAD FORM   (88% confidence, 1.1s)
```

### Dashboard display

- Total reps, good reps, bad reps — live counters
- Last rep result — large green checkmark or red X with confidence and duration
- Form score bar — green above 70%, yellow 50–70%, red below 50%
- Rep duration chart plotted over the session
- Rep history table with most recent reps first
- Session start time and live status indicator

Press **Ctrl+C** in Terminal 1 to stop the session. A summary prints to the terminal and the dashboard stops refreshing.

---

## Improving the model

To retrain with new data from additional users:

1. Record new sessions using `capture.py` with a clearly named file
2. Move CSVs into `data/good/` or `data/bad/`
3. Re-run `python train_rep_classifier.py`
4. The new `rep_classifier.pkl` overwrites the old one automatically

Adding data from multiple users significantly improves generalization since the model currently trains on one person's motion patterns.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Access is denied` on COM port | Serial Monitor is open | Close Serial Monitor in Arduino IDE |
| `A device attached is not functioning` | USB connection issue | Unplug and replug cable, try different USB port |
| `No objects to concatenate` | CSV folders empty or wrong path | Check `data/good/` and `data/bad/` have CSV files |
| `KeyError: accel_x` | Column name mismatch | Verify CSV headers match `SENSOR_COLS` in script |
| `Failed to find ICM20948 chip!` | IMU wiring issue | Check SDA to D4, SCL to D5, VIN to 3V3, GND to GND |
| Merged reps showing 5s+ duration | Prominence threshold too high | Lower `prominence` from 0.3 to 0.2 in both scripts |
| Too many false rep detections | Prominence threshold too low | Raise `prominence` from 0.3 to 0.4 in both scripts |
| Dashboard not updating | Backend not running | Make sure Terminal 1 is running before opening dashboard |
| Saved 0 rows to CSV | Arduino not streaming or ERROR state | Open Serial Monitor to verify data is flowing, then close it |

---

## Model details

- **Algorithm:** Random Forest — 200 decision trees (scikit-learn)
- **Features:** 75 per rep — mean, std, min, max, range, median, skew, kurtosis across 9 IMU axes, plus rep duration, acceleration energy, and gyro smoothness
- **Dataset:** 55 sessions (~3,000+ labeled reps), 26 good / 29 bad
- **Test accuracy:** 99.8% on held-out test reps
- **Rep detection:** Peak detection on accel_x signal with Savitzky-Golay smoothing
