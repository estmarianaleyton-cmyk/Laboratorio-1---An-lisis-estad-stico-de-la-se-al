# Laboratorio 1 - Análisis de los estadísticos descriptivos de la señal
**Universidad Militar Nueva Granada**

**Asignatura:** Procesamiento Digital de Señales

**Estudiantes:** Dubrasca Martinez, Mariana Leyton, Maria Fernanda Castellanos

**Fecha:** 22 de agosto de 2025

**Título de la practica:** Estadística de la señal

# **Objetivos**
- Identificar los estadísticos que describen una señal biomédica.
- Aplicar algoritmos de progamación en Python para importar, graficar y calcular los estadísticos descriptivos de señales fisiológicas, tanto mediante funciones programadas manualmente como con funciones predefinidas.
- Capturar señales fisiológicas con un DAQ o STM32.
- Analizar la relacion señal ruido (SNR) mediante la contaminación de las señales con diferentes tipos de ruido (gaussiano, impulso y artefacto) y evaluar su impacto en el análisis estadístico.

# **Procedimiento, método o actividades**
Este consistió en tres etapas principales, en la parte A se descargó la señal electrocardiográfica de la base de datos de [PhysioNet](https://physionet.org/), la cual se importó a Google colab para poder graficarla y posteriormente calcular cada uno de sus estadísticos descriptivos tanto de forma manual como con las funciones que contaba la librería de Numpy. En la parte B se generó una señal fisiológica del mismo tipo de la usada en la parte A utilizando un generador de señales biológicas, esta se capturo mediante un DAQ, se almaceno y posteriormente se procesó en Google colab para calcular sus estadísticos descriptivos, y compararlos con la parte A. Finalmente en la parte C, se investigó la relación señal ruido (SNR) y se contamino la señal capturada en la parte B con distintos tipos de ruido, como el Gaussiano, el de impulso y de artefacto para medir la relación señal ruido (SNR).

# **Parte A**
## **Código en Python (Google colab)**
<pre> ```
  # Importación de las librerias a utilizar
!pip install wfdb                                                    # Instalación de la liberia wfdb
import wfdb                                                          # Liberia para analizar señales fisiologicas
import matplotlib.pyplot as plt                                      # Liberia para permitir visualizar las graficas de las señales
import os                                                            # Liberia para interactuar con el sistema operativo
from google.colab import files                                       # Liberia en Google colab para subir archivos desde el computador
import numpy as np                                                   # Liberia para calculos matematicos y manejo de arreglos
from scipy.stats import kurtosis                                     # Liberia para calcular la curtosis
from scipy.stats import gaussian_kde                                 # Liberia para calcular la funcion de probabilidad y simulacion de ruido gaussiano

  # Señal ECG de apnea del sueño
if not (os.path.exists('a02.hea') and os.path.exists('a02.dat')):    # Verificar que los archivos se encuentren en colab
    print("Archivos no encontrados. Por favor, súbalos ahora:")
    uploaded = files.upload()

registro = wfdb.rdrecord('a02')                                       # Se usa la liberia wfdb para leer el nombre de los archivos, los carga y se almacenan en registro

 # Obtención de datos
señales = registro.p_signal                                           # Matriz de la señal, en donde las filas corresponden al tiempo y las columnas a los canales 
                                                                      (si el ECG tiene mas de una derivación)
fs = registro.fs                                                      # Se obtiene la frecuencia de muestreo el ECG
canal = señales[:, 0]                                                 # Elegir el primer canal (: significan todas las filas y 0 la columna 0)
tiempo = [i / fs for i in range(len(canal))]                          # Creación del vector de tiempo correspondiente a cada muestra

  # Grafica
segundos = 15                                                         # Duración de la señal que se quiere graficar
muestras = int(segundos * fs)                                         # Calculo del numero de muestras que corresponden a esos 15 segundos 

plt.figure(figsize=(10, 4))                                           # Crear las proporciones de la gráfica
plt.plot(tiempo[:muestras], canal[:muestras], label=registro.sig_name[0]) # Se dibuja la señal, en donde el tiempo corresponde al eje X y canal al eje Y
plt.title(f"ECG - primeros {segundos} s")                             # Colocar el título a la gráfica
plt.xlabel("Tiempo (s)")                                              # Colocar el nombre del eje X
plt.ylabel("Voltaje (mV)")                                            # Colocar el nombre del eje Y
plt.grid(True)                                                        # Activacion de la rejilla
plt.tight_layout()                                                    # Ajustar las margenes de la gráfica 
plt.show()                                                            # Muestra la gráfica
  ```
</pre> 
## **Gráfica de la señal ECG**
<img width="1235" height="487" alt="image" src="https://github.com/user-attachments/assets/45fa6fe4-4960-402c-86bc-61c09246a18a" />

## **Estadísticos descriptivos de la señal sin funciones de Python**
<pre> ```
print("\033[1mEstadisticos descriptivos sin funciones de python:\033[0m")
n = len(canal[:muestras])                                                 # Calcula el numero total de muestras de la señal
media = sum(canal[:muestras]) / n
suma_cuadrados = sum((x - media) **2 for x in canal[:muestras])           # Calculo de la suma de cuadrados de las desviaciones respecto a la media (hallar varianza)
desviacion = (suma_cuadrados/ n) ** 0.5                                   
coevariacion = (desviacion / media) * 100
curtosis = np.sum(((canal[:muestras]-media)/desviacion)**4)/n             # Calculo de la curtosis muestral no centrada en exceso
curtosis_F = np.sum(((canal[:muestras]-media)/desviacion)**4)/n -3        # Calculo de la curtosis de Fisher

 # Se imprimen cada uno de los resultados
print(f"\033[3mLa media de la señal es: {media:.4f} \033[0m")
print(f"\033[3mLa desviacion estandar de la señal es: {desviacion:.4f} \033[0m")
print(f"\033[3mEl coeficiente de variacion es: {coevariacion:.4f} \033[0m")
print(f"\033[3mExceso de curtosis en la señal es de: {curtosis_F:.4f} (curtosis de Fisher)\033[0m")
print(f"\033[3mLa curtosis de la señal es de: {curtosis:.4f} (curtosis muestral no centrada en exceso)\033[0m")

  # Histograma
plt.figure(figsize=(8, 4))
plt.hist(canal[:muestras], bins=30, color="purple",edgecolor="black", density=False)
plt.title("Histograma del ECG (15 s)")
plt.xlabel("Voltaje (mV)")
plt.ylabel("Frecuencia (Hz)")
plt.grid(True)
plt.show()

  # Función de probabilidad
valores_unicos = list(set(canal[:muestras]))                                             # Calcular valores únicos y sus probabilidades
probabilidades = []

for v in valores_unicos:
    frecuencia = sum(1 for x in canal[:muestras] if x == v)
    prob = frecuencia / n
    probabilidades.append((v, prob))

print("\n\033[1mFunción de probabilidad (valores únicos y su probabilidad):\033[0m")      # Imprimir los resultados
for v, p in probabilidades:
    print(f"Valor: {v:.4f}  ->  Prob: {p:.4f}")

valores = [v for v, _ in probabilidades]                                                  # Gráficar función de probabilidad
probs = [p for _, p in probabilidades]

plt.bar(valores, probs, width=0.01)  # ajusta width según los valores de tu señal
plt.xlabel("Valores de la señal")
plt.ylabel("Probabilidad")
plt.title("Función de probabilidad de la señal")
plt.show()
  ```
</pre>
## **Histograma**


## **Funcion de probabilidad**
<img width="719" height="564" alt="image" src="https://github.com/user-attachments/assets/af8b9cf6-cd75-418b-8c4e-a01fcc3c4a60" />


## **Resultados de los estadísticos descriptivos de la señal**
- *Media de la señal:* 0.0016
- *Desviacion estandar de la señal:* 0.1390
- *Coeficiente de variación:* 8778.00332
- *Exceso de curtosis (Curtosis de Fisher):* 1.7190
- *Curtosis (curtosis muestral no centrada en exceso):* 4.7190



## **Estadísticos descriptivos de la señal con funciones de Python**
<pre> ```
print("\033[1mEstadisticos descriptivos con funciones de python:\033[0m")
media = np.mean(canal[:muestras])
desviacion = np.std(canal[:muestras])
coevariacion = (desviacion / media) * 100
curtosis = kurtosis(canal[:muestras])
curtosis = kurtosis(canal[:muestras], fisher=False)
curtosis_F= kurtosis(canal[:muestras])
curtosis_F = kurtosis(canal[:muestras], fisher=True)
print(f"\033[3mLa media de la señal es: {media:.4f} \033[0m")
print(f"\033[3mLa desviacion estandar de la señal es: {desviacion:.4f} \033[0m")
print(f"\033[3mEl coeficiente de variacion es: {coevariacion:.4f} \033[0m")
print(f"\033[3mExceso de curtosis en la señal es de: {curtosis_F:.4f} (curtosis de Fisher)\033[0m")
print(f"\033[3mLa curtosis (sin exceso) de la señal es de: {curtosis:.4f} (curtosis muestral no centrada en exceso)\033[0m")

  # Histograma 
plt.figure(figsize=(8, 4))
plt.hist(canal[:muestras], bins=30, color="skyblue",edgecolor="black", density=False)
plt.title("Histograma del ECG (15 s)")
plt.xlabel("Voltaje (mV)")
plt.ylabel("Frecuencia (Hz)")
plt.grid(True)
plt.show()

  # Función de probabilidad
data = np.ravel(canal[:muestras])                        # Convertir el segmento de señal en un vector 1D ya que gaussian_kde tarabaja con datos 1D 
kde = gaussian_kde(data)                                 # Estimacion de la función de densidad de probabilidad
x_vals = np.linspace(min(data), max(data), 1000)         # Rango de valores, desde minimo hasta el maximo

plt.plot(x_vals, kde(x_vals))                            # Grafica la curva de densidad de probabilidad
plt.xlabel("Amplitud")
plt.ylabel("Densidad de probabilidad")
plt.title("Función de probabilidad del ECG (15 s)")
plt.show()
  ```
</pre>

## **Resultados de los estadísticos descriptivos de la señal**
- *Media de la señal:* 0.0016
- *Desviacion estandar de la señal:* 0.1390
- *Coeficiente de variación:* 8778.00332
- *Exceso de curtosis (Curtosis de Fisher):* 1.7190
- *Curtosis (curtosis muestral no centrada en exceso):* 4.7190

## **Histograma**
<img width="869" height="491" alt="image" src="https://github.com/user-attachments/assets/49656386-2cdd-4638-bbbc-66a823615e61" />

## **Funcion de probabilidad**
<img width="690" height="568" alt="image" src="https://github.com/user-attachments/assets/3b2103a2-68f5-4d55-9463-49f2aec8b406" />

## **Análisis de los resultados de la parte A**
El análisis de la señal ECG de 15 segundos descargada de PhysioNet permitió caracterizar su comportamiento mediante la media (0.0016 mV) cercana a cero indica que la señal está  centrada en la línea base, mientras que la desviación estándar (0.1390 mV) refleja que la mayor parte de las muestras se concentran en torno al valor central, con variaciones propias del ECG. El coeficiente de variación, resulta elevado (≈8778 %), pierde relevancia práctica debido a que la media es muy baja y el cociente se ve amplificado. En cuanto a la curtosis, los valores obtenidos (4.7190 sin exceso y 1.7190 con exceso de Fisher)  que significa que predominan eventos transitorios de alta amplitud. Estos hallazgos se respaldan en el histograma y en la función de probabilidad, que evidencian una fuerte concentración de amplitudes alrededor de cero y menor frecuencia en valores extremos. En conjunto, los resultados confirman que la señal ECG es estable, con dispersión controlada.

# **Parte B**

## **Código en Python (Google colab)**
<pre> ```
from google.colab import files
import pandas as pd
import matplotlib.pyplot as plt
uploaded = files.upload()                        # Subir los archivos a colab
df = pd.read_csv("laboratorio_datos.csv")        # Se guarda en un DataFrame de Pandas
  # Gráfica
tiempo = df.iloc[:,0].values                     # Se toma la primera columna y se guarda como vector tiempo
canal2 = df.iloc[:,0].values                     # Se toman los datos de la segunda columna y se guarda como vector canal2
plt.figure(figsize=(10, 4))
plt.plot(df["timeStamps"], df["data"], label = "ECG")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (mV)")
plt.title("Señal fisiologica tomada en el laboratorio")
plt.grid(True)
plt.show()
```
</pre>

## **Gráfica de la señal fisiológica medida en el laboratorio**
<img width="1054" height="486" alt="image" src="https://github.com/user-attachments/assets/ff285bc5-f41a-4ae8-bfea-9e0c4270b760" />

## **Estadísticos descriptivos de la señal fisiológica sin funciones de Python**
<pre> ```
print("\033[1mEstadisticos descriptivos sin funciones de python:\033[0m")
n = len(canal2)     
media = sum(canal2) / n
suma_cuadrados = sum((x - media) **2 for x in canal2)
desviacion = (suma_cuadrados / n) ** 0.5
coevariacion = (desviacion / media) * 100
n = len(canal2)
curtosis = np.sum(((canal2-media)/desviacion)**4)/n
curtosis_F = np.sum(((canal2-media)/desviacion)**4)/n -3
print(f"\033[3mLa media de la señal es: {media:.4f} \033[0m")
print(f"\033[3mLa desviacion estandar de la señal es: {desviacion:.4f} \033[0m")
print(f"\033[3mEl coeficiente de variacion es: {coevariacion:.4f} \033[0m")
print(f"\033[3mExceso de curtosis en la señal es de: {curtosis_F:.4f} (curtosis de Fisher)\033[0m")
print(f"\033[3mLa curtosis de la señal es de: {curtosis:.4f} (curtosis muestral no centrada en exceso)\033[0m")

  # Histograma
señal = df["data"]
plt.figure(figsize=(8, 4))
plt.hist(señal, bins=30, color="skyblue",edgecolor="black", density=False)
plt.title("Histograma de la señal fisiogica tomada en el laboratorio")
plt.xlabel("Voltaje (mV)")
plt.ylabel("Frecuencia (Hz)")
plt.grid(True)
plt.show()

  # Función de probabilidad discreta
senal = df["data"]
n = len(senal)
valores_unicos = list(set(senal))
probabilidades = []

for v in valores_unicos:
frecuencia = sum(1 for x in senal if x == v)
prob = frecuencia / n
probabilidades.append((v, prob))

  
valores = [v for v, _ in probabilidades]                    
probs = [p for _, p in probabilidades]

plt.figure(figsize=(10,4))
plt.bar(valores, probs, width=0.002, color="orange", edgecolor="black")
plt.xlabel("Valores de la señal (mV)")
plt.ylabel("Probabilidad")
plt.title("Función de probabilidad discreta de la señal fisiológica")
plt.grid(True)
  ```
</pre>

## **Resultados de los estadísticos descriptivos de la señal**
- *Media de la señal:* 0.2827
- *Desviacion estandar de la señal:* 0.1621
- *Coeficiente de variación:* 57.3493
- *Exceso de curtosis (Curtosis de Fisher):* 3.9548
- *Curtosis (curtosis muestral no centrada en exceso):* 6.9548

## **Histograma**
<img width="856" height="493" alt="image" src="https://github.com/user-attachments/assets/dc3a4a28-41ae-4c5e-b370-d10646a496d1" />

## **Funcion de probabilidad**
<img width="863" height="394" alt="download" src="https://github.com/user-attachments/assets/70112ae7-aef3-4584-9f14-750f5bcdd663" />


## **Estadísticos descriptivos de la señal fisiológica con funciones de Python**
<pre> ```
print("\033[1mEstadisticos descriptivos con funciones de python:\033[0m")
media = np.mean(canal2)
desviacion = np.std(canal2)
coevariacion = (desviacion / media) * 100
curtosis = kurtosis(canal2)
curtosi_F = kurtosis(canal2, fisher=False)
print(f"\033[3mLa media de la señal es: {media:.4f} \033[0m")
print(f"\033[3mLa desviacion estandar de la señal es: {desviacion:.4f} \033[0m")
print(f"\033[3mEl coeficiente de variacion es: {coevariacion:.4f} \033[0m")
print(f"\033[3mExceso de curtosis en la señal es de: {curtosis:.4f} (curtosis de Fisher)\033[0m")
print(f"\033[3mLa curtosis (sin exceso) de la señal es de: {curtosi_F:.4f} (curtosis muestral no centrada en exceso)\033[0m")

  # Histograma
señal = df["data"]
plt.figure(figsize=(8, 4))
plt.hist(senal, bins=30, color="green",edgecolor="black", density=False)
plt.title("Histograma de la señal fisiogica tomada en el laboratorio")
plt.xlabel("Voltaje (mV)")
plt.ylabel("Frecuencia (Hz)")
plt.grid(True)
plt.show()

  # Función de probabilidad
data = np.ravel(canal[:muestras])                        
kde = gaussian_kde(data)                                 
x_vals = np.linspace(min(data), max(data), 1000)        

plt.plot(x_vals, kde(x_vals))                            
plt.xlabel("Amplitud")
plt.ylabel("Densidad de probabilidad")
plt.title("Función de probabilidad del ECG (15 s)")
plt.show()
  ```
</pre>

## **Resultados de los estadísticos descriptivos de la señal**
- *Media de la señal:* 0.2827
- *Desviacion estandar de la señal:* 0.1621
- *Coeficiente de variación:* 57.3493
- *Exceso de curtosis (Curtosis de Fisher):* 3.9548
- *Curtosis (curtosis muestral no centrada en exceso):* 6.9548

## **Histograma**
<img width="860" height="490" alt="image" src="https://github.com/user-attachments/assets/fd836d2a-ea5d-45a1-a0a5-1e11c38eb90f" />

## **Funcion de probabilidad**
<img width="764" height="563" alt="image" src="https://github.com/user-attachments/assets/5a13e3ca-a9b0-4290-a4e3-cbad29548e5d" />

## **Análisis de los resultados de la parte B**
La señal adquirida mediante el DAQ se observaron cambios notorios, frente a la de PhysioNet. Ya que se adquirio directamente desde el osiloscopio, desde el cual se exporto a una memoria USB  para luego  procesarla en Pthon, al revisar los datos descargados se observo que se mostraba demasiado comprimida, lo que dificultaba su visualizacion y sus caracteristicas estadisticas, por eso fue necesario aplicar un factor de escala  (0.2x10^6=200.000), a partir de ese valor se normalizo su amplitud para asi obtener los datos. La medida de la señal tambien se mantuvo cercana a cero, lo que indica una buena adquisicion. La desviacion estandar resulto ser un poco mayor, reflejando mayor dispersion y variabilidad en los datos. 
Los resultados obtenidsos muestran que la señal adquirirda comparten caracteristicas de estaditicas similares a la de la señal de referencia.

# **Parte C**
## **Relación señal ruido (SNR)** Relaciona la señal de potencia del ruido deseado con la pontecia del ruido
**a. Contaminar la señal con ruido gaussiano y medir el SNR**
 ```
</pre>
import numpy as np
import matplotlib.pyplot as plt

def calcular_snr(signal, noisy_signal):
    potencia_signal = np.mean(signal**2)
    potencia_ruido = np.mean((noisy_signal - signal)**2)
    return 10 * np.log10(potencia_signal / potencia_ruido)

senal = df["data"].values
tiempo = df["timeStamps"].values

ruido_gauss = np.random.normal(0, 0.2, len(senal))  # media=0, sigma=0.2
senal_gauss = senal + ruido_gauss
snr_gauss = calcular_snr(senal, senal_gauss)
print(f"(a) SNR con ruido gaussiano: {snr_gauss:.2f} dB")

plt.figure(figsize=(12,4))
plt.plot(tiempo, senal, label="Señal original", alpha=0.7)
plt.plot(tiempo, senal_gauss, label="Señal con ruido gaussiano", alpha=0.7)
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (mV)")
plt.title("Señal con ruido gaussiano")
plt.legend()
plt.grid(True)
plt.show()
 ```
</pre>

<img width="1035" height="422" alt="image" src="https://github.com/user-attachments/assets/ed2eaadb-1b95-4a23-a425-25231ecdfb07" />
-En la grafica podemos apreciar que la señal ruidosa (la naranja) sigue de cerca a la original (la azul), el ruido gaussiano afecta en todo momento y por todo el rango, puesto que genera pequeñas variaciones continuas, el SNR es bastnate bajo lo que significa que la potencia de ruido es cercana a la señal, por eso se nota la distorsión.

 ```
</pre>
senal = df["data"].values
tiempo = df["timeStamps"].values

senal_impulso = senal.copy()
num_impulsos = int(0.01 * len(senal))  # 1% de los puntos alterados
posiciones = np.random.randint(0, len(senal), num_impulsos)
senal_impulso[posiciones] = senal_impulso[posiciones] + np.random.choice([3, -3], size=num_impulsos)

snr_impulso = calcular_snr(senal, senal_impulso)
print(f"(b) SNR con ruido impulso: {snr_impulso:.2f} dB")

plt.figure(figsize=(12,4))
plt.plot(tiempo, senal, label="Señal original", alpha=0.7)
plt.plot(tiempo, senal_impulso, label="Señal con ruido impulso", alpha=0.7)
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (mV)")
plt.title("Señal con ruido impulso")
plt.legend()
plt.grid(True)
plt.show()
 ```
</pre>

**b. Contaminar la señal con ruido impulso y medir el SNR**
<img width="1027" height="425" alt="image" src="https://github.com/user-attachments/assets/886522b3-eee3-42c3-a4b8-a826c9f93653" />
-El ruido de impulso (el naranja) es bastante parecido al original excepto por una parte que donde aparece un pico muy grande hacia abajo, sin embargo, esto es típico de este tipo de ruido. El SNR es extremadamente bajo, de 0.7dB, significa que esos impulsos tienen tanta energía que la potencia del ruido supera o iguala la de la señal.

senal = df["data"].values
tiempo = df["timeStamps"].values
fs = 500  # Frecuencia de muestreo aprox., cámbiala según tu archivo
t = np.arange(len(senal)) / fs

 ```
</pre>
artefacto_baja = 0.5 * np.sin(2 * np.pi * 0.5 * t)
artefacto_alta = 0.2 * np.sin(2 * np.pi * 60 * t)
senal_artefacto = senal + artefacto_baja + artefacto_alta

snr_artefacto = calcular_snr(senal, senal_artefacto)
print(f"(c) SNR con ruido tipo artefacto: {snr_artefacto:.2f} dB")

plt.figure(figsize=(12,4))
plt.plot(tiempo, senal, label="Señal original", alpha=0.7)
plt.plot(tiempo, senal_artefacto, label="Señal con artefacto", alpha=0.7)
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (mV)")
plt.title("Señal con ruido tipo artefacto")
plt.legend()
plt.grid(True)
plt.show()
 ```
</pre>

**c. Contaminar la señal con ruido tipo artefacto y medir el SNR**
<img width="1010" height="418" alt="image" src="https://github.com/user-attachments/assets/73e40d9f-9e6c-49f4-9b49-df3c436992a6" />
-La señal con el ruido tiene ondulaciones y picos que no corresponden a la señal original, aqui se mezclan interferencias sistematicas como una onda extra y el SNR es de 3.4dB lo cual es todavía bajo, indicando que los artefactos introducen una gran distorsión perceptible.



