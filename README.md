///img

### ML-LinearRegression-StudentStudyHours

///diagram

**Objetivo:**

Este proyecto se enfoca en analizar la relación entre las horas de estudio de los estudiantes y sus calificaciones. Utilizaremos regresión lineal para comprender cómo la cantidad de tiempo que un estudiante dedica al estudio influye en sus resultados académicos.

**Software utilizados:**

  - Python:
      con la librería NumPy para implementar la regresión lineal).
      con la librería Matplotlib para la creación de gráficos.
  - Jupyter Notebook para la creación y documentación del análisis.
  - Conjunto de datos "Student Study Hours" en formato CSV obtenidos de Kaggle.

**Archivos y código a crear:**

  - Un cuaderno Jupyter ("Student_Study_Hours_Analysis.ipynb") que incluya el análisis exploratorio de datos, la implementación de la regresión lineal y la visualización de resultados.
  - Un archivo CSV ("student_study_data.csv") que contiene los datos de las horas de estudio y las calificaciones de los estudiantes.

**Resultado esperado:**

Haber realizado un análisis efectivo de regresión lineal que permita comprender cómo las horas de estudio de los estudiantes se relacionan con sus calificaciones. Esto incluirá visualizaciones de datos, la evaluación del modelo de regresión y conclusiones sobre la relación entre estas variables.

***

**Resumen de los pasos a seguir:**

  - Crear un directorio o carpeta donde trabajaremos, por ejemplo "C:\ml\01_linear_regression"
  
  - Crear un nuevo archivo con extensión "ipynb": el notebook de jupyter: "Student_Study_Hours_Analysis.ipynb".
  
  - (Teniendo Docker instalado) En Windows, abrimos el programa "Docker Desktop".
      
  - Abrimos una terminal y ejecutamos el comando:

      **"docker run -p 8888:8888 -v C:\ml\01_linear_regression:/home/jovyan/work jupyter/scipy-notebook"**

    Si no tuvieramos descargada la imagen del contendedor **jupyter/scipy-notebook** Docker la descargará primero.

    ///CARBON-code
    *Unable to find image 'jupyter/scipy-notebook:latest' locally
    latest: Pulling from jupyter/scipy-notebook*

    **"docker run"**
    Comando para ejecutar un contenedor Docker a partir de una imagen (descargada previamente). Estamos iniciando un contenedor de Jupyter Notebook.

    **"-p 8888:8888"**
    Este argumento mapea el puerto del contenedor al puerto de tu máquina local. Significa que el puerto 8888 del contenedor estará disponible en tu máquina local en el puerto 8888. Esto te permitirá acceder al entorno de Jupyter Notebook desde tu navegador web.

    **"-v [ruta_local]:/home/jovyan/work"**
    Este argumento establece un volumen en Docker, que vincula una carpeta local en tu sistema a una carpeta dentro del contenedor. [ruta_local]: Debes reemplazar esto con la ubicación en tu sistema donde deseas que el contenedor acceda a los archivos. Los archivos dentro de esta ubicación en tu sistema local serán visibles dentro del contenedor en la ruta /home/jovyan/work.

    *El directorio **"/home/jovyan/work"** es un convenio común en estas imágenes para facilitar la organización de los cuadernos y archivos relacionados con tus proyectos. El usuario "jovyan" es un usuario predeterminado en muchas de estas imágenes de Jupyter, y su directorio de trabajo se establece en "/home/jovyan/work" para que los usuarios puedan acceder a él de manera sencilla.*

    **"jupyter/scipy-notebook"**
    Esto es el nombre de la imagen de Docker que se utilizará para crear el contenedor. jupyter/scipy-notebook es una imagen oficial de Jupyter Notebook que incluye bibliotecas científicas de Python.

  


    
