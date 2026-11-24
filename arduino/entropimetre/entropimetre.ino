/* Entropimetre : mesure de la variation d'entropie d'une masse d'eau.
   Livre "L'entropie sous toutes ses formes", chapitre "L'entropimetre".

   Materiel : ESP32 (carte DevKit, programmee avec l'IDE Arduino),
   sonde DS18B20 (GPIO4, resistance 4,7 kohm vers 3,3 V),
   INA219 alimente en 3,3 V (I2C : SDA=GPIO21, SCL=GPIO22),
   module MOSFET canal N pleinement passant pour 3,3 V de grille (GPIO25,
   10 kohm entre grille et masse : chauffe coupee au demarrage),
   resistance chauffante etanche 12 V / 30 W au plus (2,5 A).
   Les masses de l'ESP32 et de l'alimentation 12 V sont reliees.
   ATTENTION : jamais plus de 3,3 V sur une broche de l'ESP32.
   Bibliotheques : OneWire, DallasTemperature, Adafruit_INA219.

   Commandes serie : 'm' marche, 'a' arret, 'z' remise a zero de E et S.
   Sortie CSV : t_s,T_C,U_V,I_A,chauffe,E_J,S_JK   (E et S sans correction
   des pertes : celle-ci se fait a l'analyse).
   SECURITE : 12 V uniquement, resistance toujours immergee. */

#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_INA219.h>

const byte BROCHE_SONDE = 4;
const byte BROCHE_MOSFET = 25;
const float T_MAX = 82.0;           // coupure automatique (degres C)
const unsigned long PERIODE = 2000; // ms

OneWire bus(BROCHE_SONDE);
DallasTemperature sondes(&bus);
Adafruit_INA219 wattmetre;

bool chauffe = false;
bool ina_ok = false;
double E = 0.0;                     // energie electrique (J)
double S = 0.0;                     // somme de P dt / T (J/K)
unsigned long t_prec = 0;

void commande(bool etat) {
  chauffe = etat;
  digitalWrite(BROCHE_MOSFET, etat ? HIGH : LOW);
}

void setup() {
  pinMode(BROCHE_MOSFET, OUTPUT);
  commande(false);
  Serial.begin(115200);
  Wire.begin(21, 22);               // SDA, SCL
  sondes.begin();
  sondes.setResolution(12);         // pas de 0,0625 degre
  ina_ok = wattmetre.begin();       // calibre par defaut : 32 V, 2 A
  if (!ina_ok) Serial.println(F("# INA219 absent"));
  // Gain /8 (320 mV sur 0,1 ohm) : mesure jusqu'a 3,2 A via la tension du shunt
  Serial.println(F("t_s,T_C,U_V,I_A,chauffe,E_J,S_JK"));
  t_prec = millis();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'm' && ina_ok) commande(true);
    if (c == 'a') commande(false);
    if (c == 'z') { E = 0.0; S = 0.0; }
  }

  unsigned long t = millis();
  if (t - t_prec < PERIODE) return;

  sondes.requestTemperatures();     // environ 750 ms en 12 bits
  float T = sondes.getTempCByIndex(0);
  if (T == DEVICE_DISCONNECTED_C || T > T_MAX) commande(false);
  if (T == DEVICE_DISCONNECTED_C) {
    Serial.println(F("# sonde debranchee"));
    t_prec = t;
    return;
  }

  // Courant deduit de la tension du shunt (0,1 ohm)
  float I = wattmetre.getShuntVoltage_mV() / 100.0;  // mV / 0,1 ohm -> A
  float U = wattmetre.getBusVoltage_V();
  // Cablage : INA219 cote haut, entre le +12 V et la resistance ;
  // getBusVoltage_V = tension aux bornes de (resistance + MOSFET).
  float P = chauffe ? U * I : 0.0;

  t = millis();
  float dt = (t - t_prec) / 1000.0;
  t_prec = t;
  E += P * dt;
  S += P * dt / (T + 273.15);

  Serial.print(t / 1000.0, 1); Serial.print(',');
  Serial.print(T, 4);          Serial.print(',');
  Serial.print(U, 3);          Serial.print(',');
  Serial.print(I, 3);          Serial.print(',');
  Serial.print(chauffe ? 1 : 0); Serial.print(',');
  Serial.print(E, 1);          Serial.print(',');
  Serial.println(S, 3);
}
