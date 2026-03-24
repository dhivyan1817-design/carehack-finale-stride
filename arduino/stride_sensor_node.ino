/*
  STRIDE — Structural Risk Intelligence
  Arduino Uno Sensor Node
  
  Sensors:
    - Flex Sensor       → A0 (structural deformation)
    - Ultrasonic HC-SR04 → D9 (TRIG), D10 (ECHO) (crack/gap width)
    - MPU6050            → I2C SDA/SCL (vibration & tilt)
  
  Output format (Serial CSV, 9600 baud):
    timestamp_ms,flex_raw,ultrasonic_cm,accel_x_g,accel_y_g,accel_z_g
  
  Sampling rate: ~1 Hz
*/

#include <Wire.h>

// ── Pin definitions ────────────────────────────────────────────────────────
#define FLEX_PIN     A0
#define TRIG_PIN     9
#define ECHO_PIN     10
#define BUZZER_PIN   8

// ── MPU6050 ────────────────────────────────────────────────────────────────
#define MPU6050_ADDR  0x68
#define PWR_MGMT_1    0x6B
#define ACCEL_XOUT_H  0x3B
#define ACCEL_SCALE   16384.0   // ±2g range

// ── Thresholds ─────────────────────────────────────────────────────────────
#define FLEX_ALERT_THRESHOLD   700   // Raw ADC — tune per installation
#define VIBRATION_ALERT_G      0.35  // Total acceleration delta from 1g (gravity)

bool buzzerActive = false;

// ────────────────────────────────────────────────────────────────────────────
void setupMPU6050() {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(PWR_MGMT_1);
  Wire.write(0x00);            // Wake up MPU6050
  Wire.endTransmission(true);
}

// ────────────────────────────────────────────────────────────────────────────
void readMPU6050(float &ax, float &ay, float &az) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 6, true);

  int16_t rawX = (Wire.read() << 8) | Wire.read();
  int16_t rawY = (Wire.read() << 8) | Wire.read();
  int16_t rawZ = (Wire.read() << 8) | Wire.read();

  ax = rawX / ACCEL_SCALE;
  ay = rawY / ACCEL_SCALE;
  az = rawZ / ACCEL_SCALE;
}

// ────────────────────────────────────────────────────────────────────────────
float readUltrasonic() {
  // Send 10µs pulse
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // 30ms timeout
  if (duration == 0) return -1.0;                  // No echo
  return (duration * 0.0343) / 2.0;               // cm
}

// ────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(9600);
  Wire.begin();
  setupMPU6050();

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  // Header comment — Python reader skips lines starting with #
  Serial.println(F("# STRIDE Sensor Node — CARE HACK'26"));
  Serial.println(F("# Format: timestamp_ms,flex_raw,ultrasonic_cm,accel_x_g,accel_y_g,accel_z_g"));
  Serial.println(F("# Sampling: ~1 Hz"));

  delay(1000); // Sensor warm-up
}

// ────────────────────────────────────────────────────────────────────────────
void loop() {
  unsigned long ts = millis();

  // 1. Flex sensor (structural deformation proxy)
  int flexRaw = analogRead(FLEX_PIN);

  // 2. Ultrasonic (gap/crack width)
  float ultraCm = readUltrasonic();
  if (ultraCm < 0) ultraCm = 0.0; // Fallback

  // 3. MPU6050 (vibration / tilt)
  float ax, ay, az;
  readMPU6050(ax, ay, az);

  // Vibration magnitude (delta from gravity vector)
  float vibMag = abs(sqrt(ax*ax + ay*ay + az*az) - 1.0);

  // 4. Alert logic
  bool flexAlert     = (flexRaw > FLEX_ALERT_THRESHOLD);
  bool vibAlert      = (vibMag  > VIBRATION_ALERT_G);
  bool shouldBuzz    = flexAlert || vibAlert;

  if (shouldBuzz && !buzzerActive) {
    tone(BUZZER_PIN, 1000);  // 1kHz alert tone
    buzzerActive = true;
  } else if (!shouldBuzz && buzzerActive) {
    noTone(BUZZER_PIN);
    buzzerActive = false;
  }

  // 5. Output CSV line
  Serial.print(ts);       Serial.print(',');
  Serial.print(flexRaw);  Serial.print(',');
  Serial.print(ultraCm, 2); Serial.print(',');
  Serial.print(ax, 4);    Serial.print(',');
  Serial.print(ay, 4);    Serial.print(',');
  Serial.println(az, 4);

  delay(1000); // 1 Hz sampling
}
