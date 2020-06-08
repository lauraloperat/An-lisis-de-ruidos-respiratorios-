# -*- coding: utf-8 -*-
"""
Juan Pablo Vasco y Laura Lopera
"""
import math
import numpy as np
import librosa
import librosa.display
import scipy.signal as signal
from linearFIR import filter_design

def carga_senal(filename):
    '''
    Carga el archivo de audio y aplica filtros FIR en función de las frecuencias de interés
    
    Parameters
    ----------
    filename: Corresponde al nombre del archivo de audio .WAV que debe estar en la misma carpeta
    que el archivo Modelo.py

    Returns
    -------
    y: Corresponde a la señal original cargada
    y_bp: Corresponde a la señal entregada por el último filtro FIR
    sr: Corresponde a la frecuencia de muestreo
    '''
    y, sr = librosa.load(filename)
    fs = sr
    order, lowpass = filter_design(fs, locutoff = 0, hicutoff = 2000, revfilt = 0)
    order, highpass = filter_design(fs, locutoff = 100, hicutoff = 0, revfilt = 1)
    y_hp = signal.filtfilt(highpass, 1, y)
    y_bp = signal.filtfilt(lowpass, 1, y_hp)
    y_bp = np.asfortranarray(y_bp)
    
    return y, y_bp, sr

def filtrar(vector, valor_tipo_umbral, valor_umbral, valor_ponderado):
    '''
    Parameters
    ----------
    vector : corresponde a todos los valores de la señal del canal seleccionado.
    tipo_umbral : forma de filtrado seleccionada por el usuario (hard, soft).
    umbral :tipo de umbral seleccionada por el usuario(universal, minimax, sure).
    ponderado : forma de aplicacion del peso seleccionada por el usuario (one, SLN, MLN).

    Returns
    -------
    filtrada_reconstruida : señal final filtrada.

    '''
    
    nivel_final=np.floor(np.log2(vector.shape[0]/2) -1) #se establece cuantas veces se hace la descomposicion
    
    t_haar = decompose(vector, 1, nivel_final, []) #se llama a la funcion descomponer 
    
    thr = umbral(valor_umbral, t_haar) #se llama a la funcion umbral
    
    pesos = ponderado(valor_ponderado, t_haar) #se llama a la funcion ponderado
                 
    filtrada = tipo_umbral(valor_tipo_umbral, t_haar, pesos, thr) #se llama la funcion tipo umbral
    
    filtrada_reconstruida = rebuild(filtrada, 1, nivel_final, []) #se llama a la funcion que recompone la señal
    
    return filtrada_reconstruida


def decompose(x, nivel_actual, nivel_final, t_haar):
    '''
    Parameters
    ----------
    x : corresponde al vector que corresponde a la señal en el canal especificado.
    nivel_actual : funciona como un contador.
    nivel_final : corresponde a cuantas veces se debe hacer la descomposicion.
    t_haar : es un vector que al llamar la funcion almacena los valores de la transformada Haar.

    Returns
    -------
    Devuelve el vector que contiene los valores de la transformada Haar.

    '''

    s = np.array([(1/np.sqrt(2)), (1/np.sqrt(2))]) #definicion del valor de scale
    w = np.array([(-1/np.sqrt(2)), (1/np.sqrt(2))])   #definicion del valor de wavelet     
    
    if (nivel_actual <= nivel_final):
        if (x.shape[0] %2) != 0:
            x = np.append(x, 0) 
    #mientras nivel actual sea menor o igual a nivel final, se hace la transformada Haar
    scale_x = np.convolve(x, s, 'full')
    aprox_x = scale_x[1::2] #se submuestrea

    wavelet_x = np.convolve(x, w, 'full')
    detail_x = wavelet_x[1::2] #se submuestrea

    t_haar.append(detail_x)

    if (nivel_actual < nivel_final):
        return decompose(aprox_x, nivel_actual+1, nivel_final, t_haar)
    #mientras nivel actual sea menor a nivel final, la funcion se llama a si misma
    t_haar.append(aprox_x)
    
    return t_haar
    
def umbral(valor_umbral, vector):
    '''
    Devuelve el valor del umbral, segun lo seleccionado por el usuario (universal, minimax, sure)

    '''
    Num_samples=0 #actua como un contador
    for i in range(len(vector)):
        Num_samples = Num_samples + len(vector[i]) #se saca el numero de muestras dependiendo del detalle a analizar
    
    if valor_umbral == 0: #umbral universal
        thr = np.sqrt(2*(np.log(Num_samples)))
                
    if valor_umbral == 1: #umbral minimax
        thr = 0.3936 + 0.1829*((np.log(Num_samples))/np.log(2))
                
    if valor_umbral == 2: #umbral sure
        
        sx2=[]
        risk=[]
        for i in range(len(vector)):
            sx2 = np.append(sx2, vector[i])
            
        sx2 = np.power(np.sort(np.abs(sx2)),2)
        #se implementa la ecuacion de sure
        risk = (Num_samples-(2*np.arange(1,Num_samples + 1)) + (np.cumsum(sx2[0:Num_samples])) + np.multiply(np.arange(Num_samples,0,-1), sx2[0:Num_samples]))/Num_samples
        #Se selecciona el mejor valor como el mínimo valor del vector anterior
        best = np.min(risk)
        #Se redondea a un entero
        redondeo = int(np.round(best))
        #Se toma la raiz cuadrada del valor en la posición "best" 
        thr = np.sqrt(sx2[redondeo])
    return thr

def ponderado(valor_ponderado, vector):
    '''
    Devuelve los pesos de los detalles segun el tipo de ponderacion elegido (one,SLN, MLN)

    '''
    
    pesos = np.zeros(len(vector))
    detail1 = vector[0]
    
    if valor_ponderado == 0: #ponderado one
        pesos[:] = 1
        
    if valor_ponderado == 1: #ponderado SLN
        peso_detail1 = (np.median(np.absolute(detail1)))/0.6745
        pesos[:] = peso_detail1 #todos los detalles se multiplican por el peso del detalle 1 
    
    if valor_ponderado == 2: #ponderado MLN
        for i in range(len(vector)):
            peso_detail_x = (np.median(np.absolute(vector[i])))/0.6745
            pesos[i] = peso_detail_x #se multiplica cada detalle por su peso correspondiente
        
    return pesos

def tipo_umbral(valor_tipo_umbral, vector, pesos, thr):
    '''
  Devuelve el vector de los detalles y la aproximacion dependiendo del tipo de filtrado seleccionado (hard, soft)

    '''
    
    umbrales_definitivos = pesos*thr
    
    if valor_tipo_umbral == 0: #filtrado hard
        
        for i in range(len(vector)-1): #para recorrer el vector de transformada Haar
            for j in range(len(vector[i])): #para recorrer cada posicion de cada detalle
                if np.abs(vector[i][j]) < umbrales_definitivos[i]:
                    vector[i][j] = 0 #si el valor del detalle en esa posicion es menor que el umbral, se hace cero
                else:
                    pass #si el valor del detalle en esta posicion es mayor al umbral, permanece igual
        
    if valor_tipo_umbral == 1: #filtrado soft
        
        for i in range(len(vector)-1):#para recorrer el vector de transformada Haar
            for j in range(len(vector[i])): #para recorrer cada posicion de cada detalle
                if np.abs(vector[i][j]) < umbrales_definitivos[i]:
                    vector[i][j] = 0 #si el valor del detalle en esa posicion es menor que el umbral, se hace cero
                else: #si el valor del detalle en esta posicion es mayor al umbral, se ejecuta la ecuacion
                    sgn = np.sign(vector[i][j])
                    resta = np.abs(vector[i][j]) - umbrales_definitivos[i]
                    vector[i][j] = sgn*resta
    return vector
    
def rebuild(t_haar, nivel_actual, nivel_final, x):
    '''
    Parameters
    ----------
    t_haar : vector de transformada Haar, entregado por decompose.
    x : corresponde al vector que corresponde a la señal en el canal especificado.
    nivel_actual : funciona como un contador.
    nivel_final : corresponde a cuantas veces se debe hacer la reconstruccion.

    Returns
    -------
    Devuelve el vector de la señal reconstruida.

    '''

    s_inv = np.array([(1/np.sqrt(2)), (1/np.sqrt(2))]) #definicion del valor de scale
    w_inv = np.array([(1/np.sqrt(2)), (-1/np.sqrt(2))])#definicion del valor de wavelet

    size = len(t_haar) 
    detalle = t_haar[size - 1 - nivel_actual] #para recorrer todas las posiciones correspondientes a los detalles
    
    if (nivel_actual <= nivel_final):
        if (nivel_actual==1):
            npoints_aprox = len(t_haar[len(t_haar)-1])
            aprox_inv = np.zeros(2*npoints_aprox)
            aprox_inv[0::2] = t_haar[size-1]# se sobremuestrea          
    
        else:
            if (len(x) > len(detalle)):
                x = x[0:len(detalle)]
            npoints_aprox = len(x)
            aprox_inv = np.zeros(2*npoints_aprox)
            aprox_inv[0::2] = x #se sobremuestrea
        
        aprox_x = np.convolve(aprox_inv, s_inv, 'full') #se realiza la convolucion con scale inverso
        
        detail_inv = np.zeros(2*npoints_aprox)
        detail_inv[0::2] = detalle
        
        detail_x = np.convolve(detail_inv, w_inv, 'full') #se realiza la convolucion con wavelet inverso 
        
        x = aprox_x + detail_x

        return rebuild(t_haar, nivel_actual+1, nivel_final, x) #la funcion se llama a si misma y se aumenta el nivel actual (contador)
    
    return x

def ciclos_respiratorios(fileaudio, filetxt, sr):
    '''
    Extrae los ciclos respiratorios de una señal y sus características patológicas,
    detalladas en un archivo de texto correspondiente a la señal
    
    Parameters
    ----------
    fileaudio: Corresponde a un vector unidimensional de datos
    filetxt: Archivo de texto que detalla tiempo de inicio y fin en el audio de cada ciclo,
    además indica si hay crepitaciones y/o sibilancias en cada ciclo
    sr: Frecuencia de muestreo del audio

    Returns
    -------
    ciclos_respiratorios: Vector que contiene en su primera posición la secuencia de datos
    del ciclo, en sus segunda y tercera posición la información de crepitaciones y sibilancias
    detallada en el archivo de texto
    '''
    
    archivo_txt = np.loadtxt(filetxt) # Se carga el archivo de texto
    r, c = archivo_txt.shape # Se obtienen las filas y columnas del archivo de texto
    ciclos_respiratorios = [] # Se inicializa el vector
    
    for i in range(r): # Se recorre cada fila del archivo de texto
        
        lim_inf = math.floor(archivo_txt[i, 0]*sr) # Se lee y aproxima en muestras el tiempo de inicio del ciclo
        lim_sup = math.ceil(archivo_txt[i, 1]*sr) # Se lee y aproxima en muestras el tiempo de finalización del ciclo
        seccion = fileaudio[lim_inf:lim_sup] # Se extrae el ciclo del audio
        ciclos_respiratorios.append([seccion, archivo_txt[i, 2], archivo_txt[i, 3]]) # Se adiciona el ciclo al vector que se retorna
        
    return ciclos_respiratorios

def estadisticas(ciclo):
    '''
    Recibe una secuencia de datos y retorna algunas de sus variables estadísticas
    
    Parameters
    ----------
    ciclo: Corresponde a un vector unidimensional de datos

    Returns
    -------
    rango: El valor máximo menos el mínimo entre los datos
    varianza: La varianza de los datos
    promedio_total: La media del promedio móvil aplicado a los datos
    promedio_espectro: El promedio del energía del espectro de frecuencia de los datos
    '''
    
    rango = np.abs(np.max(ciclo) - np.min(ciclo)) # Se obtiene el rango
    varianza = np.var(ciclo) # Se obtiene la varianza
    cantidad_muestras = 1000 # Se establece el tamaño de 'ventana' para el promedio móvil
    corrimiento = 200 # Se establece el desplazamiento que hará la 'ventana' en cada iteración
    recorrido = np.arange(0, len(ciclo)-cantidad_muestras, corrimiento) # Se establece el vector que recorrerá el FOR
    promedio_local = [] # Se inicializa el vector de los promedios locales
    
    for i in recorrido:
        # Se agrega cada promedio a una posición del vector
        promedio_local.append(np.mean([ciclo[i:i+cantidad_muestras]]))
        
    promedio_local.append(np.mean([ciclo[recorrido[len(recorrido)-1]:]])) # Se agrega el promedio de la última ventana
    promedio_total = np.mean(promedio_local) # Se obtiene el promedio del vector de promedios
    f, Pxx = signal.periodogram(ciclo) # Se obtiene el espectro de frecuencia
    promedio_espectro = np.mean(Pxx) # Se promedia el espectro de frecuencia
    
    return rango, varianza, promedio_total, promedio_espectro
