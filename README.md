 # Laboratorio 1 - Análisis de los estadísticos descriptivos de la señal

**Universidad Militar Nueva Granada**

**Asignatura:** Procesamiento Digital de Señales

**Estudiantes:** Dubrasca Martinez, Mariana Leyton, Maria Fernanda Castellanos

**Fecha:** 22 de agosto de 2025

**Título de la práctica:** Estadística de la señal

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
<img width="869" height="488" alt="image" src="https://github.com/user-attachments/assets/425c9013-f739-4132-a586-6d874b6e6511" />

## **Funcion de probabilidad**
<img width="719" height="564" alt="image" src="https://github.com/user-attachments/assets/af8b9cf6-cd75-418b-8c4e-a01fcc3c4a60" />


## **Resultados de los estadísticos descriptivos de la señal**
- *Media de la señal:* 0.0016 mV
- *Desviacion estandar de la señal:* 0.1390 mV
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
- *Media de la señal:* 0.0016 mV
- *Desviacion estandar de la señal:* 0.1390 mV
- *Coeficiente de variación:* 8778.00332
- *Exceso de curtosis (Curtosis de Fisher):* 1.7190
- *Curtosis (curtosis muestral no centrada en exceso):* 4.7190

## **Histograma**
<img width="869" height="491" alt="image" src="https://github.com/user-attachments/assets/49656386-2cdd-4638-bbbc-66a823615e61" />

## **Funcion de probabilidad**
<img width="690" height="568" alt="image" src="https://github.com/user-attachments/assets/3b2103a2-68f5-4d55-9463-49f2aec8b406" />

## **Análisis de los resultados de la parte A**
El análisis de la señal ECG de 15 segundos descargada de [PhysioNet](https://physionet.org/) nos permitió entender su comportamiento a través de la media, que es de 0.0016 mV, un valor muy cercano a cero. Esto sugiere que la señal está bien centrada en la línea base, sin desplazamientos significativos. Por otro lado, la desviación estándar es de 0.1390 mV, un número pequeño que indica que la mayoría de las muestras se agrupan alrededor de un valor medio. Esto tiene sentido, ya que los intervalos entre los picos de las ondas QRS, P y T son de baja amplitud en comparación con los picos de las ondas R. El coeficiente de variación es bastante alto, alcanzando 8778.00332, debido a que la media está tan cerca de cero, lo que provoca que el valor se dispare. En cuanto a la curtosis de Fisher, el valor obtenido es de 1.7190, lo que indica que la distribución es leptocúrtica. Esto significa que tiene colas más pesadas y picos más pronunciados que una distribución normal, gracias a las ondas R que presentan una amplitud más alta. Estos hallazgos se ven respaldados por el histograma y la función de probabilidad, que muestran una fuerte concentración de amplitudes alrededor de cero y una menor frecuencia en los valores extremos.

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
canal2 = df.iloc[:,1].values                     # Se toman los datos de la segunda columna y se guarda como vector canal2
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

senal = df["data"].values

plt.figure(figsize=(15,4))

# Subplot 1: Función de probabilidad con KDE
plt.subplot(1,3,1)
sns.kdeplot(senal, fill=False, color="orange")
plt.xlabel("Valores de la señal (mV)")
plt.ylabel("Densidad de probabilidad")
plt.title("Función de probabilidad de la señal fisiológica")
plt.grid(True, linestyle="--", alpha=0.6)

  ```
</pre>

## **Resultados de los estadísticos descriptivos de la señal**
- *Media de la señal:* 0.2827 mV
- *Desviacion estandar de la señal:* 0.1621 mV
- *Coeficiente de variación:* 57.3493
- *Exceso de curtosis (Curtosis de Fisher):* 3.9548
- *Curtosis (curtosis muestral no centrada en exceso):* 6.9548

## **Histograma**
<img width="856" height="493" alt="image" src="https://github.com/user-attachments/assets/dc3a4a28-41ae-4c5e-b370-d10646a496d1" />

## **Funcion de probabilidad**
<img width="430" height="387" alt="image" src="https://github.com/user-attachments/assets/9bb73bc7-5534-4286-9c34-2315422e3513" />



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
- *Media de la señal:* 0.2827 mV
- *Desviacion estandar de la señal:* 0.1621 mV
- *Coeficiente de variación:* 57.3493
- *Exceso de curtosis (Curtosis de Fisher):* 3.9548
- *Curtosis (curtosis muestral no centrada en exceso):* 6.9548

## **Histograma**
<img width="860" height="490" alt="image" src="https://github.com/user-attachments/assets/fd836d2a-ea5d-45a1-a0a5-1e11c38eb90f" />

## **Funcion de probabilidad**
<img width="764" height="563" alt="image" src="https://github.com/user-attachments/assets/5a13e3ca-a9b0-4290-a4e3-cbad29548e5d" />

## **Análisis de los resultados de la parte B**
La señal adquirida mediante el DAQ se observaron cambios notorios, frente a la de [PhysioNet](https://physionet.org/). Ya que se capturo directamente desde el osciloscopio, desde el cual se exporto a una memoria USB para luego procesarla en Python, al revisar los datos descargados se observó que se mostraba demasiado comprimida, lo que dificultaba su visualización y calcular sus estadísticos descriptivos, por eso fue necesario aplicar un factor de escala (0.2x10^6=200.000), a partir de ese valor se normalizo su amplitud para así obtener los datos.
La señal muestra una media de 0.2827 mV, lo que indica un valor bajo y consistente con lo que esperaríamos de una señal fisiológica. Además, su desviación estándar de 0.1621 mV y un coeficiente de variación del 57.35 sugieren que hay una gran variabilidad en torno a la media, algo que se puede observar en las rápidas fluctuaciones y picos en la gráfica. Al analizar la curtosis, encontramos un exceso de 3.95 (curtosis muestral de 6.95), lo que revela una distribución leptocúrtica, caracterizada por una fuerte concentración de valores alrededor del promedio y la aparición más frecuente de picos extremos en comparación con una distribución normal. Esto se confirma en el histograma, donde la mayoría de los datos se agrupan entre 0.2 y 0.3 mV, aunque también hay valores atípicos en el rango de 0.7 a 0.8 mV y en la función de probabilidad que se obtuvo con las funciones predefinidas, muestra un comportamiento multimodal, con un primer pico entre 0.1 y 0.2 mV y un segundo aumento en las amplitudes más altas.

# **Parte C**
## **Relación señal ruido (SNR)** 
Es una medida que se usa mucho en ingeniería y comunicaciones que mide cuán fuerte es una señal útil en comparación con el ruido de fondo indeseado. Se define como la razón entre la potencia de la señal y la potencia del ruido. En la práctica, se expresa en decibelios (dB) para facilitar su interpretación en escalas logarítmicas:

<img width="298" height="87" alt="image" src="https://github.com/user-attachments/assets/98ae6fd2-a644-47a6-a7b3-40daede580a3" />

Cabe aclarar que un valor más alto de SNR indica una señal más clara y menos afectada por el ruido, lo que mejora la precisión en análisis de datos.

## **1. Contaminar la señal con ruido gaussiano y medir el SNR**

<pre> ```
def calcular_snr(signal, noisy_signal) 
    potencia_signal = np.mean(signal**2) # Potencia de la señal original = promedio de la señal al cuadrado
    potencia_ruido = np.mean((noisy_signal - signal)**2) # Potencia del ruido = promedio del error cuadrático entre señal ruidosa y origina
    return 10 * np.log10(potencia_signal / potencia_ruido) # Fórmula del SNR en decibeles (dB)
# Preparar la señal
senal = df["data"].values        # Vector con los valores de la señal original (ECG)
tiempo = df["timeStamps"].values # Vector con el eje de tiempo correspondiente
# Generar ruido gaussiano
ruido_gauss = np.random.normal(0, 0.2, len(senal))
senal_gauss = senal + ruido_gauss # Señal ruidosa = señal original + ruido gaussiano
# Calcular el SNR resultante
snr_gauss = calcular_snr(senal, senal_gauss)
print(f"(a) SNR con ruido gaussiano: {snr_gauss:.2f} dB")
# Gráfica comparativa
plt.figure(figsize=(12,4))
plt.plot(tiempo, senal, label="Señal original", alpha=0.7)          # Señal limpia
plt.plot(tiempo, senal_gauss, label="Señal con ruido gaussiano", 
         alpha=0.7)                                                 # Señal contaminada
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (mV)")
plt.title("Señal con ruido gaussiano")
plt.legend()
plt.grid(True)
plt.show()
 ```
</pre>
## **Señal con ruido gaussiano: 4.10 dB**
<img width="1235" height="474" alt="image" src="https://github.com/user-attachments/assets/2b38f327-b0ba-488e-8acc-9ca1a59d581e" />

## **Análisis de la señal con ruido gaussiano**
En la grafica podemos apreciar que la señal ruidosa (la naranja) sigue de cerca a la original (la azul), el ruido gaussiano afecta en todo momento y por todo el rango, puesto que genera pequeñas variaciones continuas, el SNR es bastnate bajo lo que significa que la potencia de ruido es cercana a la señal, por eso se nota la distorsión.

## **2. Contaminar la señal con ruido impulso y medir SNR**

<pre> ```
senal = df["data"].values 
tiempo = df["timeStamps"].values 

senal_impulso = senal.copy()
num_impulsos = int(0.01 * len(senal))   # Número de impulsos = 1% de la longitud de la señal
posiciones = np.random.randint(0, len(senal), num_impulsos)  # Posiciones aleatorias

# En esas posiciones se agregan impulsos de +3 o -3 mV
senal_impulso[posiciones] = senal_impulso[posiciones] + np.random.choice([3, -3], size=num_impulsos)

snr_impulso = calcular_snr(senal, senal_impulso)  # Cálculo del SNR
print(f"(b) SNR con ruido impulso: {snr_impulso:.2f} dB")

# Gráfica comparativa
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

## **Señal con ruido impulso: 0.72 dB**
<img width="1258" height="488" alt="image" src="https://github.com/user-attachments/assets/93f1c733-f7d3-44f4-9c5b-9c6533a4adb3" />

## **Análisis de la señal con ruido impulso**
El ruido de impulso (el naranja) es bastante parecido al original excepto por una parte que donde aparece un pico muy grande hacia abajo, sin embargo, esto es típico de este tipo de ruido. El SNR es extremadamente bajo, de 0.7dB, significa que esos impulsos tienen tanta energía que la potencia del ruido supera o iguala la de la señal.

## **3. Contaminar la señal con ruido tipo artefacto y medir SNR**

<pre> ```
senal = df["data"].values
tiempo = df["timeStamps"].values
fs = 500  # Frecuencia de muestreo (ajustar según tu archivo)
t = np.arange(len(senal)) / fs

# Artefacto de baja frecuencia (deriva de línea base, 0.5 Hz)
artefacto_baja = 0.5 * np.sin(2 * np.pi * 0.5 * t)

# Artefacto de alta frecuencia (interferencia eléctrica, 60 Hz)
artefacto_alta = 0.2 * np.sin(2 * np.pi * 60 * t)

# Señal contaminada con ambos artefactos
senal_artefacto = senal + artefacto_baja + artefacto_alta

# Cálculo del SNR
snr_artefacto = calcular_snr(senal, senal_artefacto)
print(f"(c) SNR con ruido tipo artefacto: {snr_artefacto:.2f} dB")

# Gráfica comparativa
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

## **Señal con ruido tipo artefacto: 3.41 dB**
<img width="1250" height="483" alt="image" src="https://github.com/user-attachments/assets/4228a93c-d900-41fb-b00c-ebd526d410fe" />

## **Análisis de la señal con ruido tipo artefacto**
La señal con el ruido tiene ondulaciones y picos que no corresponden a la señal original, aqui se mezclan interferencias sistematicas como una onda extra y el SNR es de 3.4dB lo cual es todavía bajo, indicando que los artefactos introducen una gran distorsión perceptible.

# **Referencias**

Academia Lab. (2025). Relación señal-ruido. Enciclopedia. Revisado el 24 de agosto del 2025. https://academia-lab.com/enciclopedia/relacion-senal-ruido/

# **Diagramas de flujo**
## **Parte A**
<img width="344" height="599" alt="image" src="https://github.com/user-attachments/assets/9582bce6-3f7f-43f0-be7c-3267f16ae3f5" />

## **Parte B**

## **Parte C**
