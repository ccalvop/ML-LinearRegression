<p align="center"><img src="https://github.com/ccalvop/ML-LinearRegression-StudentStudyHours/assets/126183973/a7d118c9-a3cb-4aee-b4c3-260ed5db1e0b" /></p>

### <p align="center">¿Qué es la regresión lineal?</p>

<p align="center">La regresión lineal es una herramienta para predecir una respuesta cuantitativa, es decir, un valor numérico continuo como una puntuación, una temperatura o un precio</p>

*En el campo de **Machine Learning**, la regresión lineal es un ejemplo de **aprendizaje supervisado**. El aprendizaje supervisado se caracteriza por utilizar un conjunto de datos etiquetados, en el que cada ejemplo de entrenamiento contiene una etiqueta o valor de salida conocido. En el caso de la regresión lineal, el objetivo es predecir un valor numérico continuo a partir de un conjunto de características de entrada. El modelo de regresión lineal aprende a ajustar una línea recta o un hiperplano (para dimensiones superiores a 2) que mejor se ajusta a los datos para realizar estas predicciones. Existen dos variantes de regresión lineal, la *regresión lineal simple* (cuando hay una sola característica de entrada) y la regresión lineal múltiple (cuando hay múltiples características de entrada).*

***

### <p align="center">Proyectos</p>

**Regresión lineal simple**:  
[ML-LinearRegression-**StudentStudyHours**](https://github.com/ccalvop/ML-LinearRegression/tree/main/StudentStudyHours)  
*En este proyecto añalizaremos la relación entre las horas de estudio de los estudiantes y sus calificaciones.  
Implementaremos dos métodos: uno manual y otro automatizado, y compararemos los resultados.*

**Regresión lineal multiple**:  
[ML-LinearRegression-**LifeExpectancy**](https://github.com/ccalvop/ML-LinearRegression/tree/main/LifeExpectancy)  
*Este proyecto busca explorar cómo diversos factores impactan en la expectativa de vida de las personas.*

***

**Objetivo:**

Emplearemos la regresión lineal simple y múltiple para analizar la conexión entre una o varias variables independientes y una variable dependiente. El objetivo es comprender de qué manera las predicciones se ven influenciadas por variaciones en la/s variable/s de entrada.

**Software y entorno utilizados:**

  - Python con diferentes librerias y dependencias (como NumPy, Pandas, Scikit-learnm, Matplolib, Seaborn...)
  - Entorno Jupyter Notebook para la creación y documentación del análisis (documento interactivo).
  - Conjunto de datos o "Dataset" en formato CSV obtenidos de sitios web como Kaggle o similar.

**Archivos y código a crear:**

  - Jupyter Notebook ("Notebook_proyecto().ipynb") que incluya el análisis de datos, la implementación de la regresión lineal y la visualización de resultados.
  - Un archivo CSV ("dataset.csv") que contiene los datos que vamos a analizar.

**Resultado esperado:**

Análisis del uso de la regresión lineal (simple o múltiple) para comprender cómo la/s variable/s independiente/s influye/n en la variable dependiente. Esto incluirá visualizaciones de datos, la evaluación del modelo de regresión y conclusiones sobre la relación entre estas variables.

***

## <p align="center">Preparación del entorno Jupyter Notebook</p>

Pasos a seguir:

  - Crear un **directorio** o carpeta donde trabajaremos en nuestra maquina local, por ejemplo en Windows "C:\ml\01_linear_regression". Copiaremos el archivo "dataset" **archivo.csv** en este directorio.
  
  - Crear un notebook con **Jupyter Notebook**:

    ***¿Qué es un notebook? ¿y Jupyter Notebook?***  
    (*)Los notebooks son documentos interactivos en los que podemos integrar texto, código ejecutable, así como, tablas o figuras.  
    (**)Jupyter Notebook es una herramienta web basada en celdas que permite programar código en Python. Es un software del Proyecto Jupyter cuyo propósito es desarrollar herramientas interactivas para Data Science y computación científica.  

    Para crear un nuevo **Jupyter Notebook** vacío que pueda abrirse correctamente, lo mejor es crearlo desde dentro del propio Jupyter Notebook. Así nos aseguraremos de que se genere un archivo .ipynb (IPythonNoteBook) con la estructura JSON adecuada y los campos necesarios para funcionar correctamente. Entonces, primero, crearemos el notebook con Jupyter.
 
  - **Jupyter Notebook**:

Podemos usar Jupyter Notebook instalado localmente (en mi caso Windows) o hacerlo funcionar desde un **contenedor Docker**: 
*La razón principal para ejecutar Jupyter Notebook desde un contenedor Docker en lugar de instalarlo localmente está relacionada con la gestión de entornos y dependencias. El uso de contenedores Docker es una práctica común en el desarrollo de software y análisis de datos, ya que proporciona un entorno controlado y aislado (con las dependencias y librerias ya instaladas).*

  - Ejecutamos **Docker**. En Windows: abrimos el programa "Docker Desktop".

    ...
    img captura windows programs docker desktop
      
  - *(Teniendo el programa "Docker Desktop" iniciado)* Abrimos una **terminal** y ejecutamos el comando:

      ```bash
      docker run -p 8888:8888 -v C:\ml\01_linear_regression:/home/jovyan/work jupyter/scipy-notebook
      ```

    Si no tuvieramos la imagen del contendedor **jupyter/scipy-notebook** descargada, Docker la descargará primero:

    ...  
    *Unable to find image 'jupyter/scipy-notebook:latest' locally
    latest: Pulling from jupyter/scipy-notebook*  
    ...

    **"docker run"** Comando para ejecutar un contenedor Docker a partir de una imagen (descargada previamente). Estamos iniciando un contenedor de Jupyter Notebook.

    **"-p 8888:8888"** Este argumento mapea el puerto del contenedor al puerto de tu máquina local. Significa que el puerto 8888 del contenedor estará disponible en tu máquina local en el puerto 8888. Esto te permitirá acceder al entorno de Jupyter Notebook desde tu navegador web.

    **"-v [ruta_local]:/home/jovyan/work"** Este argumento establece un volumen en Docker, que vincula una carpeta local en tu sistema a una carpeta dentro del contenedor. [ruta_local]: Debes reemplazar esto con la ubicación en tu sistema donde deseas que el contenedor acceda a los archivos. Los archivos dentro de esta ubicación en tu sistema local serán visibles dentro del contenedor en la ruta /home/jovyan/work.

    *El directorio **"/home/jovyan/work"** es un convenio común en estas imágenes para facilitar la organización de los cuadernos y archivos relacionados con tus proyectos. El usuario "jovyan" es un usuario predeterminado en muchas de estas imágenes de Jupyter, y su directorio de trabajo se establece en "/home/jovyan/work" para que los usuarios puedan acceder a él de manera sencilla.*

    **"jupyter/scipy-notebook"** Esto es el nombre de la imagen de Docker que se utilizará para crear el contenedor. jupyter/scipy-notebook es una imagen oficial de Jupyter Notebook que incluye bibliotecas científicas de Python.

  - Podemos observar en Docker Desktop que la imagen del contenedor se ha descargado y está en uso, y que el contenedor está corriendo (running):

    ...img
    captura docker desktop

  - En la terminal, verás un enlace que comienza con *http: //127.0.0.1:8888/?token=...* Copia este enlace y pégalo en tu navegador web. Se abrirá Jupyter Notebook en la ruta de trabajo. Podriamos crear un nuevo notebook o abrir el que creamos como archivo anteriormente.

  - Ya en Jupyter Notebook, creamos un nuevo fichero "Notebook >Python 3 (ipykernel)" y lo renombramos "Nombre_del_notebook.ipynb" para comenzar el proyecto.


TIME - 2023-12-05 13:28:13