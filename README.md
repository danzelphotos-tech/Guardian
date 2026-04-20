# Smart Workout Armband — Sensor Test

TECHIN515 Group Final Project: a wearable armband that detects exercise form during strength training and provides real-time feedback.

## Hardware

- **Microcontroller:** Seeed XIAO ESP32-S3
- **IMU:** Adafruit ICM20948 9-DoF (accelerometer, gyroscope, magnetometer, temperature)
- **Flex Sensor:** Resistive flex sensor with 47kΩ pulldown resistor (voltage divider)

## Wiring

### ICM20948 → XIAO ESP32-S3 (I2C)

| IMU Pin | XIAO Pin       |
|---------|----------------|
| VIN     | 3V3            |
| GND     | GND            |
| SDA     | D4 (GPIO5)     |
| SCL     | D5 (GPIO6)     |

### Flex Sensor → XIAO ESP32-S3 (Voltage Divider)

```
3V3 ──── [Flex Sensor] ──┬──── [47kΩ Resistor] ──── GND
                          │
                       D0 (ADC)
```

- Flex sensor pin 1 → 3V3
- Flex sensor pin 2 → breadboard junction row
- 47kΩ resistor leg 1 → same junction row
- 47kΩ resistor leg 2 → GND
- D0 wire → same junction row

## Sensor Placement

- **IMU** — worn on the wrist (captures full arc of motion during reps)
- **Flex sensor** — placed on the inside of the elbow (measures joint angle)

## Setup

### 1. Install Arduino IDE

Download from [arduino.cc](https://www.arduino.cc/en/software).

### 2. Add ESP32-S3 Board Support

1. Go to **File → Preferences**
2. In "Additional Board Manager URLs" add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. Go to **Tools → Board → Board Manager**, search `esp32`, and install
4. Select board: **XIAO_ESP32S3**

### 3. Install Libraries

In **Sketch → Include Library → Manage Libraries**, install:

- `Adafruit ICM20X`
- `Adafruit BusIO`
- `Adafruit Unified Sensor`

### 4. Upload and Run

1. Connect the XIAO ESP32-S3 via USB-C
2. Select the correct port under **Tools → Port**
3. Upload the sketch
4. Open **Serial Monitor** at **115200 baud**

## Expected Output

```
Accel X: 2.380 Y: 4.705 Z: -8.198 m/s²
Gyro  X: 0.090 Y: -0.038 Z: -0.036 rad/s
Mag   X: -1.200 Y: 17.850 Z: -61.950 uT
Temp: 28.1 °C
Flex  Raw: 0  Voltage: 0.000 V
```

### Flex Sensor Values (Bicep Curl)

| Position         | Raw Value | Voltage |
|------------------|-----------|---------|
| Fully extended   | ~0        | ~0.0 V  |
| Mid-curl         | ~1300–2800| ~1.0–2.3 V |
| Fully curled     | ~3400–3650| ~2.7–2.9 V |

## Troubleshooting

- **"Failed to find ICM20948 chip!"** — Check I2C wiring (SDA to D4, SCL to D5). Make sure VIN and GND are connected.
- **Flex sensor reads 0 constantly** — Check that the flex sensor, 47kΩ resistor, and D0 wire all share the same breadboard row at the junction.
- **Flex sensor reads ~4095 constantly** — The 47kΩ resistor is not connected to GND. Verify the resistor's second leg goes to a row wired to the XIAO's GND pin.
- **Garbled text in Serial Monitor** — Make sure baud rate is set to 115200 in the Serial Monitor dropdown.
