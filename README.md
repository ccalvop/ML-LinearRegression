<p align="center">
<img src="https://github.com/ccalvop/ML-LinearRegression-StudentStudyHours/assets/126183973/a7d118c9-a3cb-4aee-b4c3-260ed5db1e0b" />
</p>

<p align="***¿Qué es la regresión lineal?***  ">
*La regresión lineal es un ejemplo de **aprendizaje supervisado** en el campo de **machine learning**. El aprendizaje supervisado se caracteriza por utilizar un conjunto de datos etiquetado en el que cada ejemplo de entrenamiento contiene una etiqueta o valor de salida conocido. En el caso de la regresión lineal, el objetivo es predecir un valor numérico continuo (como una puntuación, una temperatura o un precio) a partir de un conjunto de características de entrada. El modelo de regresión lineal aprende a ajustar una línea recta o un hiperplano (en dimensiones superiores) que mejor se ajusta a los datos para realizar estas predicciones. Existen variantes de regresión lineal, como la *regresión lineal simple* (cuando hay una sola característica de entrada), y la regresión lineal múltiple (cuando hay múltiples características de entrada).*

**Proyectos:**

Regresión lineal simple  
ML-LinearRegression-**StudentStudyHours**  
*Este proyecto plantea analizar la relación entre las horas de estudio de los estudiantes y sus calificaciones.*

Regresión lineal multiple  
ML-LinearRegression-**LifeExpectancy**  
*Este proyecto busca explorar cómo diversos factores impactan en la expectativa de vida de las personas.*

**Objetivo:**

Emplearemos la regresión lineal simple para analizar la conexión entre una variable independiente y una variable dependiente. El objetivo es comprender de qué manera las predicciones se ven influenciadas por variaciones en la variable de entrada.

**Software y recursos utilizados:**

  - Python:
      con la librería NumPy para analizar los datos e implementar la regresión lineal.
      con la librería Matplotlib para la creación de gráficos.
  - Jupyter Notebook para la creación y documentación del análisis.
  - Conjunto de datos "Dataset" en formato CSV obtenidos de sitios web como Kaggle o similar.

**Archivos y código a crear:**

  - Jupyter Notebook ("Notebook_proyecto().ipynb") que incluya el análisis exploratorio de datos, la implementación de la regresión lineal y la visualización de resultados.
  - Un archivo CSV ("dataset.csv") que contiene los datos que vamos a analizar.

**Resultado esperado:**

Análisis efectivo de regresión lineal (simple y múltiple) que permita comprender cómo la/s variable/s independiente/s influye/n en la variable dependiente. Esto incluirá visualizaciones de datos, la evaluación del modelo de regresión y conclusiones sobre la relación entre estas variables.

***

**Preparacion del entorno Jupyter Notebook  
Pasos a seguir:**

  - Crear un directorio o carpeta donde trabajaremos, por ejemplo "C:\ml\01_linear_regression". Copiaremos el archivo "dataset" **archivo.csv** en este directorio.
  
  - **Crear un notebook con Jupyter Notebook**:

    ***¿Qué es un notebook? ¿y Jupyter Notebook?***
    (*)Los notebooks son documentos interactivos en los que podemos integrar texto, código ejecutable, así como, tablas o figuras.
    (**)Jupyter Notebook es una herramienta web basada en celdas que permite programar código en Python. Es un software del Proyecto Jupyter cuyo propósito es desarrollar herramientas interactivas para Data Science y computación científica

    Para crear un nuevo Jupyter Notebook vacío que pueda abrirse correctamente en Jupyter Notebook, lo mejor es crearlo desde dentro del propio programa web Jupyter Notebook. Así nos aseguraremos de que se genere un archivo .ipynb (IPythonNoteBook) con la estructura JSON adecuada y los campos necesarios para funcionar correctamente. Entonces, primero, crearemos el notebook con Jupyter.
 
  - **Jupyter Notebook**:

Podemos usar Jupyter Notebook instalandolo en Windows o desde un **contenedor Docker**: 
*La razón principal para ejecutar Jupyter Notebook desde un contenedor Docker en lugar de instalarlo directamente en Windows (o en otros sistemas) está relacionada con la gestión de entornos y dependencias. El uso de contenedores Docker es una práctica común en el desarrollo de software y análisis de datos, ya que proporciona un entorno controlado y aislado que facilita la gestión de dependencias y la portabilidad de proyectos.*

    (Teniendo Docker instalado) Abrimos Docker. En Windows: abrimos el programa "Docker Desktop".

    //img captura windows programs docker desktop
      
  - Abrimos una **terminal** y ejecutamos el comando:

      **"docker run -p 8888:8888 -v C:\ml\01_linear_regression:/home/jovyan/work jupyter/scipy-notebook"**

    Si no tuvieramos la imagen del contendedor **jupyter/scipy-notebook** descargada, Docker la descargará primero:

    ///CARBON-code
    *Unable to find image 'jupyter/scipy-notebook:latest' locally
    latest: Pulling from jupyter/scipy-notebook*

    **"docker run"** Comando para ejecutar un contenedor Docker a partir de una imagen (descargada previamente). Estamos iniciando un contenedor de Jupyter Notebook.

    **"-p 8888:8888"** Este argumento mapea el puerto del contenedor al puerto de tu máquina local. Significa que el puerto 8888 del contenedor estará disponible en tu máquina local en el puerto 8888. Esto te permitirá acceder al entorno de Jupyter Notebook desde tu navegador web.

    **"-v [ruta_local]:/home/jovyan/work"** Este argumento establece un volumen en Docker, que vincula una carpeta local en tu sistema a una carpeta dentro del contenedor. [ruta_local]: Debes reemplazar esto con la ubicación en tu sistema donde deseas que el contenedor acceda a los archivos. Los archivos dentro de esta ubicación en tu sistema local serán visibles dentro del contenedor en la ruta /home/jovyan/work.

    *El directorio **"/home/jovyan/work"** es un convenio común en estas imágenes para facilitar la organización de los cuadernos y archivos relacionados con tus proyectos. El usuario "jovyan" es un usuario predeterminado en muchas de estas imágenes de Jupyter, y su directorio de trabajo se establece en "/home/jovyan/work" para que los usuarios puedan acceder a él de manera sencilla.*

    **"jupyter/scipy-notebook"** Esto es el nombre de la imagen de Docker que se utilizará para crear el contenedor. jupyter/scipy-notebook es una imagen oficial de Jupyter Notebook que incluye bibliotecas científicas de Python.

  - Podemos observar en Docker Desktop que la imagen del contenedor se ha descargado y está en uso, y que el contenedor está corriendo (running):

    ///img captura docker desktop

  - En la terminal, verás un enlace que comienza con *http: //127.0.0.1:8888/?token=...* Copia este enlace y pégalo en tu navegador web. Se abrirá Jupyter Notebook en la ruta de trabajo. Podriamos crear un nuevo notebook o abrir el que creamos como archivo anteriormente.

  - Ya en Jupyter Notebook, creamos un nuevo fichero "Notebook >Python 3 (ipykernel)" y lo renombramos "Nombre_del_notebook.ipynb".

***

## ML-LinearRegression-StudentStudyHours

![diagram](https://github.com/ccalvop/ML-LinearRegression-StudentStudyHours/assets/126183973/90e9c9c6-a419-411e-a675-6ba13de42c95)

### Student_Study_Hours_Analysis.ipynb

//CARBON-code  
#imports
from numpy import *
import matplotlib.pyplot as plt

#magic command
%matplotlib inline  
//CARBON-code

**from numpy import \***   
Numpy es una biblioteca ampliamente utilizada para realizar operaciones matriciales y numéricas en Python. Importar * de numpy significa que estás importando todas las funciones y objetos de numpy en el espacio de nombres actual. Esto te permite usar funciones como np.array(), np.mean(), y otras sin tener que escribir numpy. antes de ellas.

**import matplotlib.pyplot as plt**  
Matplotlib es una biblioteca gráfica que se utiliza para crear visualizaciones, como gráficos y gráficos. matplotlib.pyplot es un módulo dentro de Matplotlib que proporciona una interfaz similar a la de MATLAB para crear gráficos. La convención de renombrar este módulo como plt es común y facilita el uso.

**%matplotlib inline**  
Es una "comando mágico" en Jupyter Notebook que permite que las gráficas generadas con Matplotlib se muestren directamente en la salida del cuaderno, en lugar de abrirse en una ventana emergente separada

//CARBON-code
#Load data
points = genfromtxt('score.csv', delimiter=',')

#Extract columns
x = array(points[:,0])
y = array(points[:,1])

#Plot the dataset
plt.scatter(x,y)
plt.xlabel('Horas de estudio')
plt.ylabel('Resultados de examenes')
plt.title('Dataset')
plt.show()
//CARBON-code

#Load data  
**points = genfromtxt('score.csv', delimiter=',')**  
Este código utiliza la función genfromtxt de la biblioteca NumPy para cargar datos desde un archivo CSV llamado 'score.csv'. El parámetro delimiter=',' indica que las columnas en el archivo CSV están separadas por comas. Los datos se cargan en la variable points, que será una matriz NumPy que contiene los datos del archivo CSV.

#Extract columns  
**x = array(points[:,0])**  
Se extrae la primera columna de la matriz points y se almacena en la variable x. 
(La expresión [:,0] se utiliza para seleccionar elementos de una matriz NumPy en función de su posición. En este caso selecciona todas las filas de la matriz y solo la primera columna)  
**y = array(points[:,1])**  
Se extrae la segunda columna de la matriz points y se almacena en la variable y

#Plot the dataset   
**plt.scatter(x,y)**
*Utiliza la función scatter de Matplotlib para crear un gráfico de dispersión de los datos. Se pasa x como los valores en el eje X (horas de estudio) y y como los valores en el eje Y (puntuaciones de los exámenes). Esto crea un gráfico que muestra cómo las puntuaciones de los exámenes están relacionadas con las horas de estudio.*  
**plt.xlabel('Horas de estudio')**  
**plt.ylabel('Resultados de examenes')**  
*Estos comandos agregan una etiqueta al eje X y al eje Y del gráfico, indicando que las unidades en el eje X son "Hours of study" (Horas de estudio) y en el eje Y son "Test scores" (Puntuaciones de los exámenes).*  
**plt.title('Dataset')**  
*Agrega un título al gráfico.*  
**plt.show()**  
*Esta función muestra el gráfico generado en la pantalla.*

//img-grafica

***

## ML-LinearRegression-2proyecto

diagrama

### 2proyecto.ipynb
