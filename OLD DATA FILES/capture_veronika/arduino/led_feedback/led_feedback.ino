/*
 * Guardian — LED Feedback Prototype (Web Serial version)
 * Hardware:
 *   - Yellow LED: D0 (GPIO1), negative to GND
 *   - Tactile button: D2 (GPIO3), other side to GND
 *
 * Flow:
 *   1. Button press → send "REP\n" to browser
 *   2. Browser classifies → sends back "GOOD\n" or "BAD\n"
 *   3. ESP32 blinks LED accordingly
 */

const int LED_PIN    = D0;
const int BUTTON_PIN = D2;

const int BLINK_ON    = 200;
const int BLINK_OFF   = 150;
const int DEBOUNCE_MS = 50;

bool lastButtonState = HIGH;
int  repCount        = 0;
bool waitingForResult = false;

void setup() {
  pinMode(LED_PIN,    OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  digitalWrite(LED_PIN, LOW);
  Serial.begin(115200);
  Serial.println("READY");
}

void loop() {
  // ── Button detection ──
  bool currentState = digitalRead(BUTTON_PIN);
  if (lastButtonState == HIGH && currentState == LOW && !waitingForResult) {
    delay(DEBOUNCE_MS);
    if (digitalRead(BUTTON_PIN) == LOW) {
      repCount++;
      waitingForResult = true;
      Serial.println("REP");
    }
  }
  lastButtonState = currentState;

  // ── Listen for result from browser ──
  if (Serial.available()) {
    String result = Serial.readStringUntil('\n');
    result.trim();
    if (result == "GOOD") {
      blinkOnce();
      waitingForResult = false;
    } else if (result == "BAD") {
      blinkTwice();
      waitingForResult = false;
    }
  }
}

void blinkOnce() {
  digitalWrite(LED_PIN, HIGH);
  delay(BLINK_ON);
  digitalWrite(LED_PIN, LOW);
}

void blinkTwice() {
  digitalWrite(LED_PIN, HIGH);
  delay(BLINK_ON);
  digitalWrite(LED_PIN, LOW);
  delay(BLINK_OFF);
  digitalWrite(LED_PIN, HIGH);
  delay(BLINK_ON);
  digitalWrite(LED_PIN, LOW);
}
