# Smart Workout Armband — Dataset Recording Guide

This guide is for anyone helping collect bicep curl training data for the Smart Workout Armband project. You do not need any machine learning experience. You just need to wear the device, do some reps, and send us the CSV files.

---

## What you need

- The Smart Workout Armband (XIAO ESP32-S3 + ICM20948 IMU) — provided
- A USB-C cable
- A laptop or desktop computer
- Python installed ([download here](https://www.python.org/downloads/) if you don't have it)

---

## One-time setup

### 1. Install the only required library

Open a terminal (Command Prompt or PowerShell on Windows, Terminal on Mac) and run:

```bash
pip install pyserial
```

That is the only install needed.

### 2. Download the files

From this repository, download two files and put them in the same folder on your computer:

- `capture.py`
- `README_recording.md` (this file)

---

## Find your COM port

This is the address your computer uses to talk to the device. You need it before recording.

**Windows:**
1. Plug in the device via USB-C
2. Open Arduino IDE (or Device Manager)
3. Go to **Tools → Port** — you will see something like `COM3`, `COM11`, `COM13`
4. Note that name

**Mac:**
1. Plug in the device via USB-C
2. Open Terminal and run: `ls /dev/cu.*`
3. Look for something like `/dev/cu.usbmodem1234` or `/dev/cu.SLAB_USBtoUART`
4. Note that name

**Linux:**
1. Plug in the device
2. Run: `ls /dev/ttyUSB*` or `ls /dev/ttyACM*`
3. Look for `/dev/ttyUSB0` or `/dev/ttyACM0`

---

## Set up capture.py

Open `capture.py` in any text editor (Notepad, TextEdit, VS Code, etc.) and find these three lines near the top:

```python
PORT = "COM13"
DURATION_SEC = 180
FILENAME = "v2_bicep_curl_ColeGood_form1.csv"
```

Change them as follows:

**PORT** — replace with your port from the step above:
```python
PORT = "COM13"             # Windows example
PORT = "/dev/cu.usbmodem1234"   # Mac example
PORT = "/dev/ttyUSB0"          # Linux example
```

**FILENAME** — rename each session with your name and what type of rep you are doing:
```python
FILENAME = "v2_bicep_curl_JaneGood_form1.csv"   # good form, session 1
FILENAME = "v2_bicep_curl_JaneBad_form1.csv"    # bad form, session 1
```

Increment the number each session: `form1`, `form2`, `form3` and so on.

**DURATION_SEC** — leave this at 180 (3 minutes per session).

---

## Wearing the device

- Strap the device to your **wrist** — not your elbow or forearm
- The sensor should sit on the **top of your wrist** (same side as your knuckles)
- It should be snug enough not to slide around during movement
- Keep the USB cable connected to your laptop while recording

---

## How to record a session

### Step 1 — Verify the device is streaming

Open Arduino IDE, go to **Tools → Serial Monitor**, set baud rate to **115200**. You should see numbers scrolling like this:

```
85098,0.611,4.223,9.041,0.119,-0.039,0.066,-6.75,27.90,7.05,30.5
85150,0.598,4.197,9.012,0.101,-0.041,0.059,-6.80,27.85,7.10,30.4
```

If you see this, the device is working. **Close Serial Monitor** before continuing — only one program can use the port at a time.

### Step 2 — Update the filename in capture.py

Change `FILENAME` to reflect what you are about to record. See naming convention below.

### Step 3 — Run capture.py

In your terminal:

```bash
cd path/to/your/folder
python capture.py
```

You will see:

```
Recording to v2_bicep_curl_JaneGood_form1.csv for 180 seconds...
Started at: 14:32:05
Start your reps NOW!
```

### Step 4 — Do your reps

Start curling immediately. Keep going for the full 3 minutes. The script will print a progress update every 100 rows so you know it is working.

### Step 5 — Wait for it to finish

After 180 seconds it stops automatically:

```
Done! Ended at: 14:35:05
Saved 3,512 rows to v2_bicep_curl_JaneGood_form1.csv
```

The CSV file is now saved in the same folder as `capture.py`.

### Step 6 — Repeat for each session

Update the filename each time and run `python capture.py` again.

---

## Naming convention

File names must follow this exact format so the training script can sort them correctly:

```
v2_bicep_curl_[YourName]Good_form[number].csv
v2_bicep_curl_[YourName]Bad_form[number].csv
```

Examples:
```
v2_bicep_curl_JaneGood_form1.csv
v2_bicep_curl_JaneGood_form2.csv
v2_bicep_curl_JaneBad_form1.csv
v2_bicep_curl_JaneBad_form2.csv
```

The words `Good` and `Bad` in the filename must be capitalized exactly as shown — the training script uses them to label your data automatically.

---

## What to record

Please record at least 5 good form and 5 bad form sessions. More is always better.

### Good form sessions
Do clean, controlled bicep curls for the full 3 minutes:
- Full range of motion — arm fully extended at the bottom, fully curled at the top
- Controlled pace — roughly 2–3 seconds per rep
- Elbow stays stationary — do not swing or use momentum
- Both arms welcome — you can alternate or do one arm at a time

### Bad form sessions
Intentionally do poor form for the full 3 minutes. Suggested bad form types (do one per session):
- **Fast/jerky curls** — curl as fast as possible, momentum-based
- **Half reps** — only curl halfway, never reaching the top
- **Shoulder raises** — raise your arm straight out in front with no elbow bend
- **Swinging** — use your whole body to swing the weight up
- **Wrist rotation only** — rotate your wrist without moving your elbow

Label each bad form session with a number so we know which session is which — you do not need to specify the type in the filename.

### Tips for clean data
- Wait 3–5 seconds after starting the script before beginning your first rep — the script trims the start automatically but giving it a moment helps
- Try to keep the band in the same position on your wrist across sessions
- If the device disconnects mid-session, the partial file is still useful — send it anyway

---

## Sending your data

Once you have recorded your sessions, send all CSV files to the project team. Any of these methods works:

- Email the CSVs directly
- Upload to the shared Google Drive folder
- Open a pull request on this repository adding your files to `data/good/` and `data/bad/`

---

## Troubleshooting

**`Access is denied` or `Permission denied` when running capture.py**
Serial Monitor is still open. Close it in Arduino IDE and try again.

**`No module named serial`**
Run `pip install pyserial` in your terminal and try again.

**`Saved 0 rows` at the end**
The device was not streaming. Open Serial Monitor first to verify data is flowing, then close it and run the script again.

**`Could not open port COM13`**
Wrong port number. Check **Tools → Port** in Arduino IDE with the device plugged in and update `PORT` in capture.py.

**Numbers look wrong or garbled in Serial Monitor**
Make sure the baud rate in Serial Monitor is set to **115200**.

**Mac — port not found**
Try running `ls /dev/cu.*` in Terminal with the device plugged in versus unplugged — the new entry that appears when plugged in is your port.

---

## Questions

Contact January or any team member if you run into issues not covered here.
