#!/usr/bin/env python
# coding: utf-8

# ## Sección 2 - Análisis estadístico
# 
# ### Laura Lopera Tobón - Juan Pablo Vasco Marín

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np


# Se procede a cargar el DataFrame generado por medio del análisis de todos los ciclos respiratorios de los audios, se corrige el nombre de algunas columnas y se eliminan algunas que han sido generadas pero no tienen información relevante para el análisis

# In[2]:


datos=pd.read_csv('ResultadosFinales.csv',header=0,sep=',')
datos.drop(['Unnamed: 1'],axis=1,inplace=True)
datos = datos.rename(columns={'Unnamed: 0': 'Archivo audio'})


# Se procede a mostrar el total de filas que posee el DataFrame, donde cada una representa un ciclo respiratorio, asociado a un determinado audio.
# Por otra parte se muestra la estructura que tiene el DataFrame mostrando sus primeras filas

# In[3]:


print("\nEl número de ciclos cargados es: ", datos.shape[0])
datos.head(n = 10)


#  **Nota:** Todas las variables estadísiticas que se extrajeron de la señal, excepto una, son acordes al análisis propuesto en las clases teóricas y talleres del curso, estas variables son sugeridas por el texto *Automated Lung Sound Analysis [2]*, en él se especifican dos métodos para obtener el promedio total, el método fino de promedio móvil y el método grueso de promedio móvil, en el presente trabajo se obtuvo mediante el método fino ya que en la sesión de clase del viernes 05 de Junio se definió que el método implementado sería a libre elección y no era necesario calcularlo por los dos métodos y calcular promedios totales diferentes.

# A continuación se muestran la descripción estadística de cada una de las variables presentes en las columnas del DataFrame cargado, implementado el comando pandas.DataFrame.describe

# In[4]:


datos.describe()


# **Nota:** En la tabla generada anteriormente se encuentra información estadística de todos los ciclos respiratorios sin ser distinguidos según las condiciones de patología. En esta tabla se puede observar para el conjunto total de datos los valores de conteo total, que coincide con el número de ciclos mostrado anteriormente, también se observa para cada columna de datos del DataFrame su promedio, su desviación estándar y sus cuartiles.

# #### Con el fin de clasificar de manera sencilla los grupos que se pretenden comparar se realiza el siguiente proceso, donde se asigna un estado específico para cada uno de los ciclos respiratorios cargados y se agrega como una nueva columna en el DataFrame. La clasificación se describe así:
# 
# - **Estado 0**: Será asignado a los ciclos respiratorios sanos, sin crepitancia ni sibilancia.
# - **Estado 1**: Será asignado a los ciclos respiratorios patológicos con sibilancia pero sin crepitancia.
# - **Estado 2**: Será asignado a los ciclos respiratorios patológicos con crepitancia pero sin sibilancia.
# - **Estado 3**: Será asignado a los ciclos respiratorios patológicos con sibilancia y crepitancia.

# In[5]:


estado = np.zeros(len(datos['Sibilancia']))

for i in range(len(datos['Sibilancia'])):
    if datos['Sibilancia'][i]==0.0 and datos['Crepitancia'][i]==0.0:
        estado[i] = 0
    elif datos['Sibilancia'][i]==1.0 and datos['Crepitancia'][i]==0.0:
        estado[i] = 1
    elif datos['Sibilancia'][i]==0.0 and datos['Crepitancia'][i]==1.0:
        estado[i] = 2
    else:
        estado[i] = 3

datos['Estado'] = estado


# A continuación se muestran algunas filas de ejemplo para mostrar el resultado de la clasificación anterior:

# In[6]:


datos.sample(n = 10)


# Adicional a esto, se hace un DataFrame temporal, el cual posee los valores promedio de cada una de las variables a analizar, discriminando el estado:

# In[7]:


df_temporal= datos[['Ciclo Respiratorio','Varianza','Rango','Promedio total','Promedio de Espectros','Estado']]
df_temporal.groupby(['Estado'],as_index=False).mean()


# Ya que se poseen los datos organizados dependiendo del estado, se procede a realizar un análisis medinte estadística descriptiva, en el cuál se observará y discutirá acerca del comportamiento de las variables y de la relación entre las mismas.

# - En primer lugar se realizan gráficos de cajas y bigotes, debido a que estos otorgan una gran cantidad de información, entre la cual se encuentran los cuartiles, la media, el valor máximo, el valor mínimo, los datos atípicos, y adicionalmente puede apreciarse la dispersión y la simetría.

# ### Diagrama de cajas y bigotes para la variable Rango en función de los estados

# In[8]:


sns.boxplot(x='Estado',y='Rango',data=datos)


# Del gráfico de cajas y bigotes correspondiente a la variable 'Rango' se puede concluir, sin tener en cuenta los datos atípicos, que todas las situaciones patológicas correspondientes a los estados 1, 2 y 3 poseen mediana, tercer cuartil y máximo superiores a los correspondientes valores que se observan para los ciclos respiratorios sanos, lo que puede hacer del rango un factor diferenciador entre estos grupos, hecho que abre la posibilidad a pre-clasificar señales de forma comparativa, entre sanos y patológicos.

# ### Diagrama de cajas y bigotes para la variable Ciclo Respiratorio en función de los estados

# In[9]:


sns.boxplot(x='Estado',y='Ciclo Respiratorio',data=datos)


# A pesar de que en este diagrama correspondiente a los ciclos respiratorios muestra diferencias respecto a la mediana, tercer cuartil y máximo entre sanos y patológicos, donde son mayores y menores respectivamente, esta información no posee validez en el análisis, esto debido a que en primera instancia es sólamente un conteo de los ciclos y hay un valor de cierta forma aleatorio para cada fila del DataFrame. Por otra parte, la cantidad de ciclos respiratorios que posee cada audio no está directamente asociado a una patología, esto se debe a que como se mencionaba en la sección uno donde se describe la base de datos utilizada, los audios contienen duraciones variadas, de manera que saber si la cantidad de respiraciones está asociada a la duración del audio o a una frecuencia respiratoria sana o patológica se convierte en una tarea imposible en este caso

# ### Diagrama de cajas y bigotes para la variable Varianza en función de los estados

# In[10]:


sns.boxplot(x='Estado',y='Varianza',data=datos)


# ### Diagrama de cajas y bigotes para la variable Promedio total en función de los estados

# In[11]:


sns.boxplot(x='Estado',y='Promedio total',data=datos)


# ### Diagrama de cajas y bigotes para la variable Promedio de Espectros en función de los estados

# In[12]:


sns.boxplot(x='Estado',y='Promedio de Espectros',data=datos)


# En estos tres últimos diagramas se encuentra gran cantidad de datos atípicos muy alejados del rango intercuartil, los cuales pueden estar relacionados al filtrado que se realizó mediante el filtro Wavelet con los vectores cuadrados Scale y Wavelet de Haar, el cual, a pesar de que logró que se presentaran mejorías con respecto a los picos de amplitud que se mostraban en la señal original y que correspondían a ruidos cardíacos, no permitió obtener una señal completamente limpia, factor que incluso puede evidenciarse al exportar y reproducir los audios de las señales filtradas, razón por la cual es posible que existan valores que no correspondan netamente a sonidos respiratorios, y que esto se vea reflejado en el análisis de las variables estadísticas.

# Posteriormente se hace uso de los histogramas, esto con el fin principal de evaluar si la distribución de los datos tienen, o no, una distribución normal, también conocida como distribución en forma de campana de Gauss, información que es crucial al momento de realizar las pruebas de hipótesis, para reconocer cuáles pruebas son válidas para ser interpretadas y cuales no, dos de las pruebas más comunes se nombran a continuación junto con la condición que les da validez:
# 
# - Prueba t: válida cuando los datos siguien una distribución normal.
# - Prueba U de Mann-Whitney: válida cuando los datos no siguien una distribución normal.

# ### Histograma para la variable Estado

# In[13]:


count, bin_edges= np.histogram(datos['Estado'])
datos['Estado'].plot(kind='hist',xticks=bin_edges)


# Mediante este gráfico puede concluirse que en los estudios realizados se obtuvo una mayor cantidad de ciclos respiratorios normales, seguido de ciclos que presentaron crepitaciones, luego ciclos con aparición de sibilancias, y finalmente, una minoría de ciclos que tuvieron ambos sonidos respiratorios patológicos, todo esto traducido al sistema de nomenclatura de estados establecido en orden de mayor a menor es: 0, 2, 1, 3.

# ### Histograma para la variable Rango

# In[14]:


count, bin_edges= np.histogram(datos['Rango'])
datos['Rango'].plot(kind='hist',xticks=bin_edges)


# ### Histograma para la variable Varianza

# In[15]:


count, bin_edges= np.histogram(datos['Varianza'])
datos['Varianza'].plot(kind='hist',xticks=bin_edges)


# ### Histograma para la variable Promedio total

# In[16]:


count, bin_edges= np.histogram(datos['Promedio total'])
datos['Promedio total'].plot(kind='hist',xticks=bin_edges)


# ### Histograma para la variable Promedio de Espectros

# In[17]:


count, bin_edges= np.histogram(datos['Promedio de Espectros'])
datos['Promedio de Espectros'].plot(kind='hist',xticks=bin_edges)


# In[18]:


count, bin_edges= np.histogram(datos['Ciclo Respiratorio'])
datos['Ciclo Respiratorio'].plot(kind='hist',xticks=bin_edges)


# Al analizar el comportamiento y la distribución de las variables, se puede observar que ninguna posee una distribución normal, es decir, ninguna sigue la tendecia de la campana de Gauss que corresponde a una función simétrica, en donde el máximo punto corresponde al centro de la función y toma el valor de la media de los datos.
# 
# La única variable que a simple vista pareciera que podría estar distribuída de manera normal, es el promedio total o promedio móvil; pero al analizar las principales características de este tipo de distribución, se evidencia que no se comporta de manera simétrica, y además lo que correspondería al pico de la campana no concuerda con el valor promedio que se arrojó anteriormente al usar el comando *describe*.

# 
# Adicionalmente, se ejecuta la matriz de correlaciones, la cual asocia todas las variables entre sí, arrojando valores entre -1 y 1, donde -1 implica que el comportamiento de las variables es opuesto, lo que quiere decir que si en un tramo, una de las dos es creciente la otra será decrecerá, y viceversa, por otra parte, un valor de 1 indica que ambas poseen una tendencia igual de incremento y disminución. Y para examinar este comportamiento, se remite a la correlación lineal, la cual se lleva a cabo únicamente entre dos variables de interés, donde la magnitud total del valor arrojado va entre 0 y 1, siendo 1 el máximo de correspondencia entre los cambios en las variables y 0 para una correspondencia nula, es decir, que el comportamiento de las variables no está relacionado de ninguna manera.
# 

# ### Matriz de correlaciones entre las variables

# In[19]:


correlation_matrix=datos.corr()
sns.heatmap(correlation_matrix,annot=True)


# ##### Con el fin de esclarecer los valores arrojados en la matriz anterior y el concepto de correlación explicado en el párrafo anterior, se muestran algunos ejemplos y un caso atípico que se ha presentado en la matriz

# ### Curva de correlación alta y positiva entre Varianza y Rango

# In[20]:


sns.regplot(x='Varianza',y='Rango',data=datos)


# ### Curva de correlación baja y negativa entre Varianza y Promedio total

# In[21]:


sns.regplot(x='Varianza',y='Promedio total',data=datos)


# ### Curva de correlación máxima y positiva entre Varianza y Promedio de Espectros

# In[22]:


sns.regplot(x='Varianza',y='Promedio de Espectros',data=datos)


# Como se puede observar, el resultado numérico que arroja la matriz de correlación está ligado a la pendiente de una curva de tendencia de correlación, se encuentran valores próximos a la magnitud uno y positivos como es el caso de la primera curva presentada, sin embargo, también pueden presentarse curvas de tendencia de magnitudes cercanas a uno pero negativas, lo que indica que tienen un comportamiento muy similar pero inverso.
# 
# Por otra parte, se observa el caso de la segunda curva, cuya pendiente es de un valor muy cercano a cero como se observa en la matriz, por este motivo se ve que el cambio en el orden de mangnitud de la varianza no está relacionado con un cambio apreciable en el promedio total a pesar de que este último está en un orden de mangnitud mucho menor. Aun así, se encuentra que en este bajo nivel de correlación cuando uno aumenta el otro parece disminuir, por lo que se obtiene un signo negativo.
# 
# Finalmente, no se puede dejar pasar por alto el resultado atípico que se obtiene para la correlación entre la Varianza y el Promedio de Espectros. Este resultado resulta inesperado debido a que usualmente se encuentran valores de correlación igual a 1 en aquellas celdas de la matriz en que coincide una variable consigo misma, como se puede apreciar en la diagonal de la matriz, sin embargo, se obtuvo este valor para dos variables que no sólo son diferentes, sino que una proviene de los datos en el dominio del tiempo y la otra en el dominio de la frecuencia. Dicho resultado es motivo de apertura de una discusión y análisis profundo del sentido que tiene este valor de correlación, sin embargo, en el presente trabajo será únicamente apreciado como un valor de correlación positivo y máximo atípico.

# ## Pruebas de hipótesis

# Finalmente, se procede a ejecutar las pruebas de hipótesis, en las cuales se plantean dos hipótesis contradictorias, una denominada la hipótesis nula, que es la que se quiere rechazar y generalmente su enunciado tiene relación con una ausencia de diferencias entre los datos, y otra llamada la hipótesis alternativa, que es la que se desea constatar o determinar como verdadera. Para tomar la decisión de rechazar alguna de las dos hipótesis, es necesario tener en cuenta un parámetro denominado *valor p*, el cual define si un valor encontrado puede deberse o no a la casualidad, y este se compara con el nivel de significancia o *alpha*, el cual representa la probabilidad de cometer un error como: rechazar la hipótesis nula cuando es verdadera, o no rechazar la hipótesis nula cuando esta es falsa. Resulta importante resaltar que dependiendo del número de comparaciones a realizar, el valor de alpha cambia, debido a que el error tiende a incrementarse, siendo inicialmente 0.05 para comparar dos grupos mediante una sola variable, pero si se tienen *n* comparaciones para hacer entre esos mismos dos grupos, se tendrá que *alpha=(0.05/n)*, y para poder rechazar la hipótesis nula, el valor p debe estar por debajo del correspondiente al nuevo *alpha*. 
# 
# 
# Para realizar este proceso, se crean 4 nuevos DataFrames, que se separan dependiendo de los resultados de los ciclos.

# ### DataFrame de los ciclos respiratorios Sanos

# In[23]:


Sanos = datos[datos['Estado'].isin([0])]
Sanos.head()


# ### DataFrame de los ciclos respiratorios con Sibilancia

# In[24]:


Sibilancia = datos[datos['Estado'].isin([1])]
Sibilancia.head()


# ### DataFrame de los ciclos respiratorios con Crepitancia

# In[25]:


Crepitancia = datos[datos['Estado'].isin([2])]
Crepitancia.head()


# ### DataFrame de los ciclos respiratorios con Sibilancia y Crepitancia

# In[26]:


Ambos = datos[datos['Estado'].isin([3])]
Ambos.head()


# Como se dijo anteriormente, se comprobó que ninguna de las variables que se poseen, se comporta de manera normal, por lo tanto no se puede recurrir a una prueba paramétrica como lo es la prueba t, sino que debe emplearse una no paramétrica, en este caso la **U de Mann-Whitney para pruebas no pareadas**, en la cual los datos no deben tener una distribución especial. Se analizarán diferentes casos, y en cada uno de ellos se ejecutarán 4 comparaciones, por lo que **el valor final de *alpha* es de 0.0125**, valor encontrado luego de seguir la ecuación mencionada anteriormente para cuando se tienen n comparaciones.

# A continuación se presenta entonces las divisiones de los grupos a evaluar y cada una de sus respectivas hipótesis, su prueba y la conclusión respecto a las hipótesis según el valor p respecto al valor alfa. Las hipótesis nulas se reconocerán con **Hn** y las hipótesis alternativas se nombrarán como **Ha**:

# In[27]:


from scipy.stats import ttest_ind, mannwhitneyu
alpha = 0.0125


# ## Sanos contra sibilancia

# #### Hipótesis respecto al Rango
# 
# - **Hn**: los ciclos respiratorios que presentan sibilancias no muestran diferencias significativas en el rango, al comparar con los ciclos normales.
# - **Ha**: los ciclos respiratorios con presencia de sibilancias son diferenciables en el rango, de los ciclos normales.

# In[28]:


statistics, pvalue = mannwhitneyu(Sanos['Rango'], Sibilancia['Rango'])
print(pvalue <= alpha)


# #### Resultado
# - Se puede rechazar la hipótesis nula.

# #### Hipótesis respecto a la Varianza
# 
# - **Hn**: los ciclos respiratorios que presentan sibilancias no muestran diferencias significativas en la varianza, al comparar con los ciclos normales.
# - **Ha**: los ciclos respiratorios con presencia de sibilancias son diferenciables en la varianza, de los ciclos normales.

# In[29]:


statistics, pvalues= mannwhitneyu(Sanos['Varianza'], Sibilancia['Varianza'])
print(pvalues <= alpha)


# #### Resultado
# - Se puede rechazar la hipótesis nula.

# #### Hipótesis respecto al Promedio total
# 
# - **Hn**: los ciclos respiratorios que presentan sibilancias no muestran diferencias significativas en la media de los promedios móviles, al comparar con los ciclos normales.
# - **Ha**: los ciclos respiratorios con presencia de sibilancias son diferenciables en la media de los promedios móviles, de los ciclos normales.

# In[30]:


statistics, pvalues= mannwhitneyu(Sanos['Promedio total'], Sibilancia['Promedio total'])
print(pvalues <= alpha)


# #### Resultado
# - No se puede rechazar la hipótesis nula.

# #### Hipótesis respecto al Promedio de Espectros
# 
# - **Hn**: los ciclos respiratorios que presentan sibilancias no muestran diferencias significativas en el promedio de los espectros de frecuencias, al comparar con los ciclos normales.
# - **Ha**: los ciclos respiratorios con presencia de sibilancias son diferenciables en el promedio de los espectros de frecuencias, de los ciclos normales.

# In[31]:


statistics, pvalues= mannwhitneyu(Sanos['Promedio de Espectros'], Sibilancia['Promedio de Espectros'])
print(pvalues <= alpha)


# #### Resultado
# - Se puede rechazar la hipótesis nula.

# ## Sanos contra crepitancia

# #### Hipótesis respecto al Rango
# 
# - **Hn**: los ciclos respiratorios que presentan crepitaciones no muestran diferencias significativas en el rango, al comparar con los ciclos normales.
# - **Ha**: los ciclos respiratorios con presencia de crepitaciones son diferenciables, en el rango de los ciclos normales.

# In[32]:


statistics, pvalues= mannwhitneyu(Sanos['Rango'], Crepitancia['Rango'])
print(pvalues <= alpha)


# #### Resultado
# - Se puede rechazar la hipótesis nula.

# #### Hipótesis respecto a la Varianza
# 
# - **Hn**: los ciclos respiratorios que presentan crepitaciones no muestran diferencias significativas en la varianza, al comparar con los ciclos normales.
# - **Ha**: los ciclos respiratorios con presencia de crepitaciones son diferenciables en la varianza, de los ciclos normales.

# In[33]:


statistics, pvalues= mannwhitneyu(Sanos['Varianza'], Crepitancia['Varianza'])
print(pvalues <= alpha)


# #### Resultado
# - Se puede rechazar la hipótesis nula.

# #### Hipótesis respecto al Promedio total
# 
# - **Hn**: los ciclos respiratorios que presentan crepitaciones no muestran diferencias significativas en la media de los promedios móviles, al comparar con los ciclos normales.
# - **Ha**: los ciclos respiratorios con presencia de crepitaciones son diferenciables en la media de los promedios móviles, de los ciclos normales.

# In[34]:


statistics, pvalues= mannwhitneyu(Sanos['Promedio total'], Crepitancia['Promedio total'])
print(pvalues <= alpha)


# #### Resultado
# - No se puede rechazar la hipótesis nula.

# #### Hipótesis respecto al Promedio de Espectros
# 
# - **Hn**: los ciclos respiratorios que presentan crepitaciones no muestran diferencias significativas en el promedio de los espectros de frecuencias, al comparar con los ciclos normales.
# - **Ha**: los ciclos respiratorios con presencia de crepitaciones son diferenciables en el promedio de los espectros de frecuencias, de los ciclos normales.

# In[35]:


statistics, pvalues= mannwhitneyu(Sanos['Promedio de Espectros'], Crepitancia['Promedio de Espectros'])
print(pvalues <= alpha)


# #### Resultado
# - Se puede rechazar la hipótesis nula.

# ### Sanos contra Crepitancia-Sibilancia

# #### Hipótesis respecto al Rango
# 
# - **Hn**: los ciclos respiratorios que presentan sibilancias y crepitaciones no muestran diferencias significativas en el rango, al comparar con los ciclos normales.
# - **Ha**: los ciclos respiratorios con presencia de sibilancias y crepitaciones son diferenciables en el rango, de los ciclos normales.

# In[36]:


statistics, pvalues= mannwhitneyu(Sanos['Rango'], Ambos['Rango'])
print(pvalues <= alpha)


# #### Resultado
# - Se puede rechazar la hipótesis nula.

# #### Hipótesis respecto a la Varianza
# 
# - **Hn**: los ciclos respiratorios que presentan sibilancias y crepitaciones no muestran diferencias significativas en la varianza, al comparar con los ciclos normales.
# - **Ha**: los ciclos respiratorios con presencia de sibilancias y crepitaciones son diferenciables en la varianza, de los ciclos normales.

# In[37]:


statistics, pvalues= mannwhitneyu(Sanos['Varianza'], Ambos['Varianza'])
print(pvalues <= alpha)


# #### Resultado
# - Se puede rechazar la hipótesis nula.

# #### Hipótesis respecto al Promedio total
# 
# - **Hn**: los ciclos respiratorios que presentan sibilancias y crepitaciones no muestran diferencias significativas en la media de los promedios móviles, al comparar con los ciclos normales.
# - **Ha**: los ciclos respiratorios con presencia de sibilancias y crepitaciones son diferenciables en la media de los promedios móviles, de los ciclos normales.

# In[38]:


statistics, pvalues= mannwhitneyu(Sanos['Promedio total'], Ambos['Promedio total'])
print(pvalues <= alpha)


# #### Resultado
# - No se puede rechazar la hipótesis nula.

# #### Hipótesis respecto al Promedio de Espectros
# 
# - **Hn**: los ciclos respiratorios que presentan sibilancias y crepitaciones no muestran diferencias significativas en el promedio de los espectros de frecuencias, al comparar con los ciclos normales.
# - **Ha**: los ciclos respiratorios con presencia de sibilancias y crepitaciones son diferenciables en el promedio de los espectros de frecuencias, de los ciclos normales.

# In[39]:


statistics, pvalues= mannwhitneyu(Sanos['Promedio de Espectros'], Ambos['Promedio de Espectros'])
print(pvalues <= alpha)


# #### Resultado
# - Se puede rechazar la hipótesis nula.

# ## Conclusiones
# 
# - Respecto a las pruebas de hipótesis se puede concluir que efectivamente existen factores entre los ciclos respitaratorios normales y los ciclos respiratorios patológicos que permiten su diferenciación mediante software, sin embargo, no son todos los factores estadísticos, únicamente se puede diferenciar estos grupos por el comportamiento de las variables: Rango, Varianza y Promedio de Espectros
# 
# 
# - Respecto al proceso de análisis de estadística descriptiva y el análisis de las pruebas de hipótesis se puede concluir que son un procedimiento de gran valor en cuando al desarrollo de tecnologías que permitan apoyar en el diagnóstico de diferentes enfermedades como son las respiratorias, debido a que permiten saber qué características de la información que se posee ayudan a diferenciar condiciones patológicas respecto a las condiciones normales
# 
# 
# - Se concluye que la existencia de grandes bases de datos de información médica, con sus respectivas etiquetas e información detallada de los fenómenos implicados, de los pacientes o sujetos evaluados y de los diagnósticos entregados por los médicos especializados, permiten el desarrollo acelarado de las nuevas tecnologías del software como son todas las relacionadas con el Machine Learning, al entregar una información valiosa para el entrenamiento de sistemas de aprendizaje profundo y redes neuronales.

# ### Referencias
# 
# - [1] Shi, Lukui & Du, Kang & Zhang, Chaozong & Ma, Hongqi & Yan, Wenjie. "Lung Sound Recognition Algorithm Based on VGGish-BiGRU", 2019
# - [2] M. Grønnesby, “Automated Lung Sound Analysis”, 2016.
