
#include <ArduinoBLE.h>
int TimeA = 4000;
// variables for button
const int buttonPin = 2;
int oldButtonState = LOW;

int command = 3;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // configure the button pin as input
  pinMode(buttonPin, INPUT);

  // initialize the Bluetooth® Low Energy hardware
  BLE.begin();

  Serial.println("Bluetooth® Low Energy Central - LED control");

  // start scanning for peripherals
  BLE.scanForUuid("19b10000-e8f2-537e-4f6c-d104768a1214");
}

void loop() {
  // check if a peripheral has been discovered
  BLEDevice peripheral = BLE.available();

  if (peripheral) {
    // discovered a peripheral, print out address, local name, and advertised service
    Serial.print("Found ");
    Serial.print(peripheral.address());
    Serial.print(" '");
    Serial.print(peripheral.localName());
    Serial.print("' ");
    Serial.print(peripheral.advertisedServiceUuid());
    Serial.println();

    if (peripheral.localName() != "Pleuro") {
      return;
    }

    // stop scanning
    BLE.stopScan();

    controlLed(peripheral);

    // peripheral disconnected, start scanning again
    BLE.scanForUuid("19b10000-e8f2-537e-4f6c-d104768a1214");
  }
}

void controlLed(BLEDevice peripheral) {
  // connect to the peripheral
  Serial.println("Connecting ...");

  if (peripheral.connect()) {
    Serial.println("Connected");
  } else {
    Serial.println("Failed to connect!");
    return;
  }

  // discover peripheral attributes
  Serial.println("Discovering attributes ...");
  if (peripheral.discoverAttributes()) {
    Serial.println("Attributes discovered");
  } else {
    Serial.println("Attribute discovery failed!");
    peripheral.disconnect();
    return;
  }

  // retrieve the LED characteristic
  BLECharacteristic ledCharacteristic = peripheral.characteristic("19b10001-e8f2-537e-4f6c-d104768a1214");

  if (!ledCharacteristic) {
    Serial.println("Peripheral does not have LED characteristic!");
    peripheral.disconnect();
    return;
  } else if (!ledCharacteristic.canWrite()) {
    Serial.println("Peripheral does not have a writable LED characteristic!");
    peripheral.disconnect();
    return;
  }

  while (peripheral.connected()) {
      if (Serial.available()>0){
        int command = Serial.read();
        Serial.println("This is the command 1");
        Serial.println(command);
      if (command == 50) {
        Serial.println("Sending Command 2 Long Sweep");
        ledCharacteristic.writeValue((byte)0x02);
        Serial.println("Left");
        delay(TimeA);
      } 
      else if (command == 51) {
        Serial.println("Sending Command 3 Medium Sweep");
         for (int ii = 0; ii<= 14; ii++) {
        ledCharacteristic.writeValue((byte)0x04);
        Serial.println("Left");
        delay(TimeA);
        ledCharacteristic.writeValue((byte)0x05);
        delay(TimeA);
        Serial.println("Right");
      }
      }
      else if (command == 52) {
      Serial.println("Sending Command 4 Short Sweep");
      ledCharacteristic.writeValue((byte)0x04);
      //}
      }
      else if (command == 53) {
      Serial.println("Sending Command 5");
      ledCharacteristic.writeValue((byte)0x05); 
      }
      else if (command == 54) {
      Serial.println("Sending Command 6");
      ledCharacteristic.writeValue((byte)0x06); 
      }
      else if (command == 55) {
      Serial.println("Sending Command 7");
      ledCharacteristic.writeValue((byte)0x07); 
      }
      else if (command == 56) {
      Serial.println("Sending Command 8");
      ledCharacteristic.writeValue((byte)0x08); 
      }
      else if (command == 57) {
      Serial.println("Sending Command 9");
      ledCharacteristic.writeValue((byte)0x09); 
      }
      else if (command == 65) {
      Serial.println("Sending Command A");
      ledCharacteristic.writeValue((byte)0x10); 
      }
      else if (command == 66) {
      Serial.println("Sending Command B");
      ledCharacteristic.writeValue((byte)0x11); 
      }
      else if (command == 67) {
      Serial.println("Sending Command C");
      ledCharacteristic.writeValue((byte)0x12); 
      }
      else {
        Serial.println("button released");

        ledCharacteristic.writeValue((byte)0x00);
      }
      }
    }

  Serial.println("Peripheral disconnected");
}
