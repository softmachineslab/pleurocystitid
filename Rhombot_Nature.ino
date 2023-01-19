int SR = 2;
int SL = 5;
int LBU = 9;
int LBD = 10;
int SBU = 8;
int SBD = 7;
int dutyR40 = 200;
int dutyL40 = 200;
int dutyR30 = 70;
int dutyL30 = 70;
int dutyR20 = 70;
int dutyL20 = 70;
int dutyS = 0;
int TimeA = 4000;

#include <ArduinoBLE.h>

BLEService ledService("19B10000-E8F2-537E-4F6C-D104768A1214"); // Bluetooth® Low Energy LED Service

// Bluetooth® Low Energy LED Switch Characteristic - custom 128-bit UUID, read and writable by central
BLEByteCharacteristic switchCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite);

const int ledPin = LED_BUILTIN; // pin to use for the LED

void setup() {
  //Serial.begin(9600);
  pinMode(SR, OUTPUT);
  pinMode(SL, OUTPUT);
  pinMode(LBD, OUTPUT);
  pinMode(LBU, OUTPUT);
  pinMode(SBU, OUTPUT);
  pinMode(SBD, OUTPUT);
  //while (!Serial);

  // set LED pin to output mode
  pinMode(ledPin, OUTPUT);

  // begin initialization
  if (!BLE.begin()) {
    //Serial.println("starting Bluetooth® Low Energy module failed!");

    while (1);
  }

  // set advertised local name and service UUID:
  BLE.setLocalName("Pleuro");
  BLE.setAdvertisedService(ledService);

  // add the characteristic to the service
  ledService.addCharacteristic(switchCharacteristic);

  // add service
  BLE.addService(ledService);

  // set the initial value for the characeristic:
  switchCharacteristic.writeValue(0);

  // start advertising
  BLE.advertise();

  //Serial.println("BLE LED Peripheral");
}

void loop() {
  // listen for Bluetooth® Low Energy peripherals to connect:
  BLEDevice central = BLE.central();

  // if a central is connected to peripheral:
  if (central) {

    // while the central is still connected to peripheral:
    while (central.connected()) {
      if (switchCharacteristic.written()) {
        int BLESig = switchCharacteristic.value();
        if (BLESig) {   // any value other than 0
          
          //Serial.println(BLESig);
          
          if (BLESig == 2){
            analogWrite(SL, (dutyL40));
            delay(TimeA);
            analogWrite(SL, 0);
          }
          else if (BLESig == 3){
            analogWrite(SR, dutyR40);
            delay(TimeA);
            analogWrite(SR, 0);
          }
          else if (BLESig == 4){
            for (int ii = 0; ii <= 14; ii++){
             analogWrite(SL, dutyL40);
             analogWrite(SR, 0);
             delay(TimeA);
             analogWrite(SL,0);
             analogWrite(SR, dutyR40);
             delay(TimeA);  
            }
            analogWrite(SR, 0);
            analogWrite(SL, 0); 
          }
          else if (BLESig == 5){
            analogWrite(SL, dutyL30);
            delay(TimeA);
            analogWrite(SL, 0);
          }
          else if (BLESig == 6){
            analogWrite(SR, dutyR20);
            delay(TimeA);
            analogWrite(SR, 0);
          }
          else if (BLESig == 7){
            analogWrite(SL, dutyL20);
            delay(TimeA);
            analogWrite(SL, 0);
          }
          else if (BLESig == 8){
            analogWrite(SL, dutyS);
            delay(5000);
            analogWrite(SL, 0);
          }

          else {
            analogWrite(SBD, 0);
            analogWrite(LBD, 0);
            analogWrite(SL, 0);
            analogWrite(SBU, 0);
            analogWrite(LBU, 0);
            analogWrite(SR, 0);
          }
          
          digitalWrite(ledPin, HIGH);         
          analogWrite(SBD, 0);
            analogWrite(LBD, 0);
            analogWrite(SL, 0);
            analogWrite(SBU, 0);
            analogWrite(LBU, 0);
            analogWrite(SR, 0);
        } else {                              
          digitalWrite(ledPin, LOW);          
          analogWrite(SBD, 0);
            analogWrite(LBD, 0);
            analogWrite(SL, 0);
            analogWrite(SBU, 0);
            analogWrite(LBU, 0);
            analogWrite(SR, 0);
        }
      }
    }
  }
}
