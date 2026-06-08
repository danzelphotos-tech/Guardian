#include <Adafruit_ICM20X.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <Adafruit_NeoPixel.h>

#define SERVICE_UUID        "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_UUID "abcdefab-cdef-abcd-efab-cdefabcdefab"
#define DEVICE_NAME         "WorkoutArmband"
#define LED_PIN             D0
#define LED_COUNT           1
#define BRIGHTNESS          40

Adafruit_NeoPixel pixel(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_ICM20948 icm;
BLECharacteristic* pCharacteristic = nullptr;
bool deviceConnected = false;
unsigned long startTime;

class ConnectionCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    pixel.setPixelColor(0, pixel.Color(0, 200, 0));
    pixel.show();
  }
  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    pixel.setPixelColor(0, pixel.Color(255, 80, 0));
    pixel.show();
    pServer->startAdvertising();
  }
};

void setup() {
  delay(1000);

  Serial.begin(115200);

  pixel.begin();
  pixel.setBrightness(BRIGHTNESS);
  pixel.setPixelColor(0, pixel.Color(0, 0, 0));
  pixel.show();

  Wire.begin(5, 6);

  if (!icm.begin_I2C()) {
    Serial.println("ERROR: Failed to find ICM20948 chip!");
    while (1) {
      pixel.setPixelColor(0, pixel.Color(255, 0, 0));
      pixel.show();
      delay(200);
      pixel.setPixelColor(0, pixel.Color(0, 0, 0));
      pixel.show();
      delay(200);
    }
  }

  icm.setAccelRange(ICM20948_ACCEL_RANGE_4_G);
  icm.setGyroRange(ICM20948_GYRO_RANGE_500_DPS);
  icm.setMagDataRate(AK09916_MAG_DATARATE_10_HZ);
  Serial.println("ICM20948 ready");

  BLEDevice::init(DEVICE_NAME);
  BLEServer* pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ConnectionCallbacks());
  BLEService* pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pCharacteristic->addDescriptor(new BLE2902());

  pService->start();
  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->start();

  Serial.println("BLE advertising as 'WorkoutArmband' — waiting for connection...");

  pixel.setPixelColor(0, pixel.Color(255, 80, 0));
  pixel.show();

  startTime = millis();
}

void loop() {
  if (!deviceConnected) {
    delay(100);
    return;
  }

  sensors_event_t accel, gyro, mag, temp;
  icm.getEvent(&accel, &gyro, &temp, &mag);

  unsigned long elapsed = millis() - startTime;
  String line = String(elapsed) + "," +
                String(accel.acceleration.x, 3) + "," +
                String(accel.acceleration.y, 3) + "," +
                String(accel.acceleration.z, 3) + "," +
                String(gyro.gyro.x, 3) + "," +
                String(gyro.gyro.y, 3) + "," +
                String(gyro.gyro.z, 3) + "," +
                String(mag.magnetic.x, 3) + "," +
                String(mag.magnetic.y, 3) + "," +
                String(mag.magnetic.z, 3) + "," +
                String(temp.temperature, 1);

  pCharacteristic->setValue(line.c_str());
  pCharacteristic->notify();

  Serial.println(line);

  delay(50);
}
