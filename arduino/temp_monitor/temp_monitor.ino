/*
  Corneal storage temperature monitor - starter sketch
  Sensor: TMP117 (I2C)
  Serial output: one temperature in Celsius per line at 9600 baud

  Install "Adafruit TMP117" from Arduino IDE Library Manager.
  LCD support is intentionally left optional because I2C LCD addresses and
  modules differ. Test the sensor and serial output first.
*/

#include <Wire.h>
#include <Adafruit_TMP117.h>
#include <Adafruit_Sensor.h>

Adafruit_TMP117 tmp117;

void setup() {
  Serial.begin(9600);
  Wire.begin();

  if (!tmp117.begin()) {
    Serial.println("TMP117_NOT_FOUND");
    while (1) {
      delay(100);
    }
  }
}

void loop() {
  sensors_event_t temp;
  sensors_event_t voltage;
  tmp117.getEvent(&temp, &voltage);

  // Stream a numeric Celsius reading for the Python dashboard.
  Serial.println(temp.temperature, 2);
  delay(1000);
}
