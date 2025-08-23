# Laboratorio 1 - Análisis de los estadísticos descriptivos de la señal
**Universidad Militar Nueva Granada**

**Asignatura:** Procesamiento Digital de Señales

**Estudiantes:** Dubrasca Martinez, Mariana Leyton, Maria Fernanda Castellanos

**Fecha:** 22 de agosto de 2025

**Titulo de la practica:** Estadística de la señal

# **Objetivos**
- Identificar los estadísticos que describen una señal biomédica.
- Obtenerlos a partir de algoritmos de programación y mostrarlos
- ddddd
# Procedimiento, método o actividades
Se descargo una señal electrocardiografica de la base de datos de PhysioNet, esta se importo a Google colab para poder graficarla y posteriormente calcular cada uno de sus estadisticos descrptivos.
# Parte A
## Código en Python (Google colab)
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

  #s Señal ECG de apnea del sueño
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

## **Análisis de la señal**


## **Estadísticos descriptivos de la señal sin funciones de python**
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
# Calcular valores únicos y sus probabilidades
valores_unicos = list(set(canal[:muestras]))
probabilidades = []

for v in valores_unicos:
    frecuencia = sum(1 for x in canal[:muestras] if x == v)
    prob = frecuencia / n
    probabilidades.append((v, prob))

# --- Mostrar resultados ---
print("\n\033[1mFunción de probabilidad (valores únicos y su probabilidad):\033[0m")
for v, p in probabilidades:
    print(f"Valor: {v:.4f}  ->  Prob: {p:.4f}")

# --- Graficar ---
valores = [v for v, _ in probabilidades]
probs = [p for _, p in probabilidades]

plt.bar(valores, probs, width=0.01)  # ajusta width según los valores de tu señal
plt.xlabel("Valores de la señal")
plt.ylabel("Probabilidad")
plt.title("Función de probabilidad de la señal")
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
<img width="865" height="492" alt="image" src="https://github.com/user-attachments/assets/86673820-4f9e-48cd-ab81-f598e625be68" />

## **Funcion de probabilidad**
<img width="719" height="564" alt="image" src="https://github.com/user-attachments/assets/af8b9cf6-cd75-418b-8c4e-a01fcc3c4a60" />


