/* Module Peltier : generateur (Seebeck) et pompe a chaleur.
   Livre "L'entropie sous toutes ses formes", chapitre Onsager.
   ESP32, deux sondes DS18B20 sur GPIO4 (4,7 kohm vers 3,3 V),
   INA219 (SDA=GPIO21, SCL=GPIO22, alimente en 3,3 V).
   Sonde 0 : bloc d'aluminium ; sonde 1 : radiateur (a verifier en chauffant
   une sonde a la main : l'ordre depend de leurs adresses).
   Aucune commande : le courant (1 A) vient d'une alimentation de laboratoire
   a limitation de courant ; on inverse le courant en permutant les fils.
   Sortie CSV : t_s,Tbloc_C,Trad_C,U_V,I_A
   L'INA219 ne mesure pas de tension negative : le placer du cote du
   "plus" de l'alimentation, avant l'inverseur. */

#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_INA219.h>

OneWire bus(4);
DallasTemperature sondes(&bus);
Adafruit_INA219 ina;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);               // SDA, SCL
  sondes.begin();
  sondes.setResolution(12);
  if (!ina.begin()) Serial.println(F("# INA219 absent"));
  Serial.println(F("t_s,Tbloc_C,Trad_C,U_V,I_A"));
}

void loop() {
  sondes.requestTemperatures();     // environ 750 ms
  Serial.print(millis() / 1000.0, 1); Serial.print(',');
  Serial.print(sondes.getTempCByIndex(0), 4); Serial.print(',');
  Serial.print(sondes.getTempCByIndex(1), 4); Serial.print(',');
  Serial.print(ina.getBusVoltage_V(), 3); Serial.print(',');
  Serial.println(ina.getShuntVoltage_mV() / 100.0, 3);  // shunt 0,1 ohm
  delay(200);
}
