# Analisis-de-sonidos-respiratorios
Este algoritmo extrae los archivos de audio de la base de datos 'Respiratory Sound Database' de Kaggle, en donde tambiénm hay presencia de un archivo de texto por cada audio, que posee una descripción de los inicios y finales de ciclos respiratorios identificados previamente, y si hay presencia de ruidos patológicos como las crepitaciones y las sibilancias. Luego de esta extracción, se ejecuta un filtro FIR con las frecuencias de interés y posteriormente una eliminación de ruido cardíaco mediante Wavelet con vectores de Haar. Adicionalmente se crea un DataFrame con variables estadísticas como la varianza, el rango, el promedio de los espectros de frecuencia y la media de los promedios móviles, para cada ciclo descrito en cada uno de los archivos.
Adicionalmente se realiza un análisis estadístico a partir de la estadística descriptiva y de las pruebas de hipótesis, con el fin de encontrar las semejanzas y diferencias entre los ciclos normales y los patológicos.
