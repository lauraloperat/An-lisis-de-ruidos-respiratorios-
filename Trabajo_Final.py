# -*- coding: utf-8 -*-
"""
Juan Pablo Vasco y Laura Lopera
"""
# Se importan las librerías necesarias
import pandas as pd
import librosa
import librosa.display
from Modelo import carga_senal, filtrar, estadisticas, ciclos_respiratorios
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'qt') #Es línea permite sacar los plt.plot en una ventana externa a la consola

# Se indica la dirección donde están almacenados los archivos de audio y texto y se crea un arreglo con todos los
# nombres de los archivos por separado según su formato
directorio='C://Users//juanv//Desktop//Trabajo_Final//Respiratory_Sound_Database//audio_and_txt_files'
archivos_audio = [file for file in listdir(directorio) if file.endswith(".wav") if isfile(join(directorio, file))]
archivos_texto = [file for file in listdir(directorio) if file.endswith(".txt") if isfile(join(directorio, file))]

# Muestra de ejemplos de aplicación de preprocesamiento
# Archivo 101_1b1_Al_sc_Meditron.wav
audio_original1, audio_filtrado1, sr = carga_senal("101_1b1_Al_sc_Meditron.wav")
ruido_corazon1 = filtrar(audio_filtrado1, 0, 1, 2)
senal_definitiva1 = audio_filtrado1 - ruido_corazon1[0:len(audio_filtrado1)]
# Archivo 180_1b4_Pr_mc_AKGC417L.wav
audio_original2, audio_filtrado2, sr = carga_senal("180_1b4_Pr_mc_AKGC417L.wav")
ruido_corazon2 = filtrar(audio_filtrado2, 0, 1, 2)
senal_definitiva2 = audio_filtrado2 - ruido_corazon2[0:len(audio_filtrado2)]

# Se generan las gráficas necesarias para ilustrar el proceso de acondicionamiento
# Se muestran las dos señales ejemplo originales 
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(audio_original1)
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.title("Archivo 101_1b1_Al_sc_Meditron.wav")
plt.subplot(2,1,2)
plt.plot(audio_original2)
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.title("Archivo 180_1b4_Pr_mc_AKGC417L.wav")
plt.show()

# Se muestra el resultado de aplicar los filtros FIR
plt.figure(2)
plt.plot(audio_original1)
plt.plot(audio_filtrado1)
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.title("Archivo 101_1b1_Al_sc_Meditron.wav")
plt.legend(['Señal original','Señal filtrada con filtros FIR'])
plt.show()

plt.figure(3)
plt.plot(audio_original2)
plt.plot(audio_filtrado2)
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.title("Archivo 180_1b4_Pr_mc_AKGC417L.wav")
plt.legend(['Señal original','Señal filtrada con filtros FIR'])
plt.show()

# Se muestra el ruido del corazón extraido de la señal con el método wavelet
plt.figure(4)
plt.subplot(2,1,1)
plt.plot(ruido_corazon1)
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.title("Ruido de corazón en el archivo 101_1b1_Al_sc_Meditron.wav")
plt.subplot(2,1,2)
plt.plot(ruido_corazon2)
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.title("Ruido de corazón en el archivo 180_1b4_Pr_mc_AKGC417L.wav")
plt.show()

# Se muestra la señal final resultante del acondicionamiento: Filtrado con filtros FIR y
# extracción del ruido de corazón entregado por el método wavelet
plt.figure(5)
plt.plot(audio_filtrado1)
plt.plot(senal_definitiva1)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Señal definitiva del archivo 101_1b1_Al_sc_Meditron.wav')
plt.legend(['Señal filtrada con filtros FIR','Señal acondicionada definitiva'])
plt.show()

plt.figure(6)
plt.plot(audio_filtrado2)
plt.plot(senal_definitiva2)
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.title('Señal definitiva del archivo 180_1b4_Pr_mc_AKGC417L.wav')
plt.legend(['Señal filtrada con filtros FIR','Señal acondicionada definitiva'])
plt.show()

# Se limpian e inicializan las variables necesarias para usarlas en el procesamiento y
# análisis de todos los arhivos que contiene la base de datos
audio_original = [] 
audio_filtrado = []
sr = 0
ruido_corazon = []
senal_definitiva = []
frame_global = []

for i in range(len(archivos_audio)): #Se itera el mismo número de veces que cantidad de archivos existentes
    
      print('Analisis #' + str(i+1) + ' ' + archivos_audio[i])
      # Se aplican filtros FIR
      audio_original, audio_filtrado, sr = carga_senal(archivos_audio[i]) 
      # Se obtiene el ruido del corazón con wavelet y parámetros:
          # Tipo de umbral: HARD
          # Umbral: Minimax
          # Ponderación: MLN
      ruido_corazon = filtrar(audio_filtrado, 0, 1, 2)
      # Se sustrae el ruido del corazón de la señal entregada por los filtros FIR
      senal_definitiva = audio_filtrado - ruido_corazon[0:len(audio_filtrado)]
     
      #Se extraen los ciclos respiratorios y sus detalles según la información de su respectivo archivo de texto
      info_audio = ciclos_respiratorios(senal_definitiva, archivos_texto[i], sr)
      # Se crea el encabezado del DataFrame del archivo de texto actual
      data_frame_audio = pd.DataFrame(columns=['Ciclo Respiratorio', 'Crepitancia', 'Sibilancia', 'Rango', 'Varianza', 'Promedio total', 'Promedio de Espectros'])
     
      #Se analiza cada ciclo retornado por la función ciclos respiratorios y se concatena cada uno en el DataFrame
      for datos_ciclo in range(len(info_audio)):
         
          ciclo = float(datos_ciclo + 1)
          crepitancia = info_audio[datos_ciclo][1]
          sibilancia = info_audio[datos_ciclo][2]
          rango, varianza, promedio_total, promedio_espectro = estadisticas(info_audio[datos_ciclo][0])
          data_frame_ciclo = pd.DataFrame({'Ciclo Respiratorio':[ciclo], 'Crepitancia':[crepitancia], 'Sibilancia':[sibilancia], 'Rango':[rango], 'Varianza':[varianza], 'Promedio total':[promedio_total], 'Promedio de Espectros':[promedio_espectro]})
          data_frame_audio = pd.concat([data_frame_audio, data_frame_ciclo], ignore_index=True)
      
      # Se agrega el DataFrame del audio analizado al DataFrame global que contendrá todos los ciclos
      frame_global.append(data_frame_audio)

# Se concatena el DataFrame global con los títulos de los archivos en un DataFrame final
Final_DataFrame = pd.concat(frame_global, keys = archivos_audio)

# Se exporta el DataFrame en formato .CSV
Final_DataFrame.to_csv('ResultadosFinales.csv')        
