<p align="center"><img src="https://github.com/ccalvop/ML-LinearRegression-StudentStudyHours/assets/126183973/a7d118c9-a3cb-4aee-b4c3-260ed5db1e0b" /></p>

#### <p align="center">¿Qué es la regresión lineal?</p>

<p align="center">La regresión lineal es una herramienta útil para predecir una respuesta cuantitativa</p>

*La regresión lineal es un ejemplo de **aprendizaje supervisado** en el campo de **machine learning**. El aprendizaje supervisado se caracteriza por utilizar un conjunto de datos etiquetados, en el que cada ejemplo de entrenamiento contiene una etiqueta o valor de salida conocido. En el caso de la regresión lineal, el objetivo es predecir un valor numérico continuo (como una puntuación, una temperatura o un precio) a partir de un conjunto de características de entrada. El modelo de regresión lineal aprende a ajustar una línea recta o un hiperplano (para dimensiones superiores a 2D) que mejor se ajusta a los datos para realizar estas predicciones. Existen dos variantes de regresión lineal, la *regresión lineal simple* (cuando hay una sola característica de entrada) y la regresión lineal múltiple (cuando hay múltiples características de entrada).*

#### <p align="center">Proyectos</p>

**Regresión lineal simple**:  
ML-LinearRegression-**StudentStudyHours**. *Este proyecto plantea analizar la relación entre las horas de estudio de los estudiantes y sus calificaciones.*

**Regresión lineal multiple**:  
ML-LinearRegression-**LifeExpectancy**. *Este proyecto busca explorar cómo diversos factores impactan en la expectativa de vida de las personas.*

***

**Objetivo:**

Emplearemos la regresión lineal simple y múltiple para analizar la conexión entre una o varias variables independientes y una variable dependiente. El objetivo es comprender de qué manera las predicciones se ven influenciadas por variaciones en la/s variable/s de entrada.

**Software y recursos utilizados:**

  - Python:
      con la librería NumPy para analizar los datos e implementar la regresión lineal.
      con la librería Matplotlib para la creación de gráficos.
  - Jupyter Notebook para la creación y documentación del análisis (documento interactivo).
  - Conjunto de datos o "Dataset" en formato CSV obtenidos de sitios web como Kaggle o similar.

**Archivos y código a crear:**

  - Jupyter Notebook ("Notebook_proyecto().ipynb") que incluya el análisis exploratorio de datos, la implementación de la regresión lineal y la visualización de resultados.
  - Un archivo CSV ("dataset.csv") que contiene los datos que vamos a analizar.

**Resultado esperado:**

Análisis efectivo de regresión lineal (simple o múltiple) que permita comprender cómo la/s variable/s independiente/s influye/n en la variable dependiente. Esto incluirá visualizaciones de datos, la evaluación del modelo de regresión y conclusiones sobre la relación entre estas variables.

***

### <p align="center">Preparación del entorno Jupyter Notebook</p>

Pasos a seguir:

  - Crear un **directorio** o carpeta donde trabajaremos en nuestra maquina local, por ejemplo en Windows "C:\ml\01_linear_regression". Copiaremos el archivo "dataset" **archivo.csv** en este directorio.
  
  - Crear un notebook con **Jupyter Notebook**:

    ***¿Qué es un notebook? ¿y Jupyter Notebook?***
    (*)Los notebooks son documentos interactivos en los que podemos integrar texto, código ejecutable, así como, tablas o figuras.
    (**)Jupyter Notebook es una herramienta web basada en celdas que permite programar código en Python. Es un software del Proyecto Jupyter cuyo propósito es desarrollar herramientas interactivas para Data Science y computación científica

    Para crear un nuevo **Jupyter Notebook** vacío que pueda abrirse correctamente, lo mejor es crearlo desde dentro del propio Jupyter Notebook. Así nos aseguraremos de que se genere un archivo .ipynb (IPythonNoteBook) con la estructura JSON adecuada y los campos necesarios para funcionar correctamente. Entonces, primero, crearemos el notebook con Jupyter.
 
  - **Jupyter Notebook**:

Podemos usar Jupyter Notebook instalado localmente (en mi caso Windows) o hacerlo funcionar desde un **contenedor Docker**: 
*La razón principal para ejecutar Jupyter Notebook desde un contenedor Docker en lugar de instalarlo localmente está relacionada con la gestión de entornos y dependencias. El uso de contenedores Docker es una práctica común en el desarrollo de software y análisis de datos, ya que proporciona un entorno controlado y aislado (con las dependencias y librerias ya instaladas).*

    (Teniendo Docker instalado) Abrimos Docker. En Windows: abrimos el programa "Docker Desktop".

    >>>/img captura windows programs docker desktop
      
  - *(En Windows, teniendo el programa "Docker Desktop" iniciado)* Abrimos una **terminal** y ejecutamos el comando:

      **"docker run -p 8888:8888 -v C:\ml\01_linear_regression:/home/jovyan/work jupyter/scipy-notebook"**

    Si no tuvieramos la imagen del contendedor **jupyter/scipy-notebook** descargada, Docker la descargará primero:

    >>>/CARBON-code
    *Unable to find image 'jupyter/scipy-notebook:latest' locally
    latest: Pulling from jupyter/scipy-notebook*
    >>>/CARBON-code

    **"docker run"** Comando para ejecutar un contenedor Docker a partir de una imagen (descargada previamente). Estamos iniciando un contenedor de Jupyter Notebook.

    **"-p 8888:8888"** Este argumento mapea el puerto del contenedor al puerto de tu máquina local. Significa que el puerto 8888 del contenedor estará disponible en tu máquina local en el puerto 8888. Esto te permitirá acceder al entorno de Jupyter Notebook desde tu navegador web.

    **"-v [ruta_local]:/home/jovyan/work"** Este argumento establece un volumen en Docker, que vincula una carpeta local en tu sistema a una carpeta dentro del contenedor. [ruta_local]: Debes reemplazar esto con la ubicación en tu sistema donde deseas que el contenedor acceda a los archivos. Los archivos dentro de esta ubicación en tu sistema local serán visibles dentro del contenedor en la ruta /home/jovyan/work.

    *El directorio **"/home/jovyan/work"** es un convenio común en estas imágenes para facilitar la organización de los cuadernos y archivos relacionados con tus proyectos. El usuario "jovyan" es un usuario predeterminado en muchas de estas imágenes de Jupyter, y su directorio de trabajo se establece en "/home/jovyan/work" para que los usuarios puedan acceder a él de manera sencilla.*

    **"jupyter/scipy-notebook"** Esto es el nombre de la imagen de Docker que se utilizará para crear el contenedor. jupyter/scipy-notebook es una imagen oficial de Jupyter Notebook que incluye bibliotecas científicas de Python.

  - Podemos observar en Docker Desktop que la imagen del contenedor se ha descargado y está en uso, y que el contenedor está corriendo (running):

    >>>/img captura docker desktop

  - En la terminal, verás un enlace que comienza con *http: //127.0.0.1:8888/?token=...* Copia este enlace y pégalo en tu navegador web. Se abrirá Jupyter Notebook en la ruta de trabajo. Podriamos crear un nuevo notebook o abrir el que creamos como archivo anteriormente.

  - Ya en Jupyter Notebook, creamos un nuevo fichero "Notebook >Python 3 (ipykernel)" y lo renombramos "Nombre_del_notebook.ipynb".

***

## ML-LinearRegression-StudentStudyHours

![diagram](https://github.com/ccalvop/ML-LinearRegression-StudentStudyHours/assets/126183973/90e9c9c6-a419-411e-a675-6ba13de42c95)

### Student_Study_Hours_Analysis.ipynb

>>>/CARBON-code  
#imports
from numpy import *
import matplotlib.pyplot as plt

#magic command
%matplotlib inline  
>>>/CARBON-code

**from numpy import \***   
Numpy es una biblioteca ampliamente utilizada para realizar operaciones matriciales y numéricas en Python. Importar * de numpy significa que estás importando todas las funciones y objetos de numpy en el espacio de nombres actual. Esto te permite usar funciones como np.array(), np.mean(), y otras sin tener que escribir numpy. antes de ellas.

**import matplotlib.pyplot as plt**  
Matplotlib es una biblioteca gráfica que se utiliza para crear visualizaciones, como gráficos y gráficos. matplotlib.pyplot es un módulo dentro de Matplotlib que proporciona una interfaz similar a la de MATLAB para crear gráficos. La convención de renombrar este módulo como plt es común y facilita el uso.

**%matplotlib inline**  
Es una "comando mágico" en Jupyter Notebook que permite que las gráficas generadas con Matplotlib se muestren directamente en la salida del cuaderno, en lugar de abrirse en una ventana emergente separada

...
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
...

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

>>>/CARBON-code
#Hyperparameters
learning_rate = 0.01
initial_b = 0
initial_m = 0
num_iterations = 50
>>>/CARBON-code

Los hiperparámetros son configuraciones que afectan el proceso de entrenamiento del modelo. Los hiperparámetros específicos de la regresión lineal simple son:

**learning_rate**: La tasa de aprendizaje determina qué tan grande es el paso que se toma en la dirección del gradiente.

**initial_b e initial_m**: Son los valores iniciales para los coeficientes de la regresión.

**num_iterations:** Indica el número de iteraciones de entrenamiento.

*En el repositorio **()** podemos ver otro ejemplo de aprendizaje supervisado: XGBoost. Los hiperparámetros que se han configurado en el ejemplo de XGBoost son específicos de ese algoritmo. Cada algoritmo tiene sus propios hiperparámetros, y estos pueden tener nombres diferentes y servir para propósitos distintos.*  
*(XGBoost es un algoritmo de aprendizaje automático supervisado que se utiliza para tareas de clasificación y regresión. El nombre "XGBoost" proviene de "eXtreme Gradient Boosting". Es una implementación eficiente y escalable de árboles de decisión potenciados por gradiente. Este algoritmo tiene sus propios hiperparámetros que son relevantes para su funcionamiento óptimo. max_depth: Controla la profundidad máxima de los árboles. eta: Es el learning rate, que determina cuánto se ajustan los pesos del modelo en cada iteración. gamma: Parámetro que controla la ganancia mínima necesaria para hacer una partición adicional en un nodo hoja del árbol. min_child_weight: Es el mínimo peso necesario de una hoja del árbol. subsample: Controla la fracción de las instancias que se utilizarán para entrenar cada árbol. objective: Define el objetivo del aprendizaje (en este caso, binary:logistic indica una tarea de clasificación binaria). num_round: Indica el número de iteraciones de entrenamiento.)*

#Hyperparameters  
**learning_rate**  
Este es un hiperparámetro que controla el tamaño de los pasos que el algoritmo de optimización toma durante el entrenamiento. Un valor más bajo hace que los pasos sean más pequeños, lo que puede conducir a una convergencia más precisa, pero también puede hacer que el entrenamiento sea más lento. Un valor demasiado alto puede hacer que el entrenamiento diverja o salte sobre el mínimo óptimo.  
*Un valor de 0.01 es bajo. Apropiado para comenzar, evita oscilaciones excesivas en el proceso de entrenamiento.*  
**initial_b**  
Este es el valor inicial del término independiente (también conocido como sesgo o intercept). En una regresión lineal, es el valor de y cuando x es 0. Al inicio del entrenamiento, el modelo utiliza este valor como punto de partida. 
*Un valor de 0 representa la estimación inicial de la variable dependiente cuando todas las variables independientes son cero.*  
**initial_m**  
Este es el valor inicial de la pendiente de la línea. Representa la tasa de cambio de la variable dependiente respecto a la independiente. Al igual que initial_b, este valor se utiliza como punto de partida en el entrenamiento.  
*Un valor de 0. Inicializarlo en 0 significa que al principio no hay pendiente. No hay un cambio inicial en la variable dependiente en relación con las variables independientes.*  
**num_iterations** Este hiperparámetro determina cuántas veces el algoritmo de entrenamiento recorre todo el conjunto de datos de entrenamiento para ajustar los parámetros del modelo. Cada pasada a través del conjunto de datos se conoce como una iteración. Un número mayor de iteraciones puede permitir que el modelo converja mejor, pero también puede aumentar el tiempo de entrenamiento.  
*Un valor de 50. El modelo debería converger relativamente rápido. Es importante monitorear la curva de aprendizaje para asegurarse de que el modelo no esté sobreajustando o subajustando los datos.*  

>>>/CARBON-code
#Cost function 
def compute_cost(b, m, points):
    total_cost = 0
    
    # Extracting the x and y values from the points array
    x = points[:, 0]  # Extracting the x values (feature)
    y = points[:, 1]  # Extracting the y values (target)
    
    # Compute sum of squared errors
    total_cost = np.sum((y - (m * x + b)) ** 2)
    
    # Return average of squared error
    return total_cost / len(points)
>>>/CARBON-code

La función **compute_cost** tiene como propósito calcular el costo o la pérdida del modelo en función de los parámetros b (independiente) y m (pendiente). El objetivo del entrenamiento es encontrar los valores de b y m que minimizan este costo. La función de costo (o función de pérdida) en el contexto de la regresión lineal es una medida de cuánto se desvían las predicciones del modelo de los valores reales. En una regresión lineal, la función de costo comúnmente utilizada es el error cuadrático medio (MSE, por sus siglas en inglés). El objetivo del entrenamiento es minimizar esta función.

**def compute_cost(b, m, points)**  
*Toma tres argumentos: b, m, y points. b y m son los parámetros del modelo que queremos evaluar, mientras que points es el conjunto de datos sobre el cual estamos calculando el costo.*  
**total_cost = 0**  
*Crea e inicializa la variable total_cost a 0. Esta variable se utilizará para acumular el costo.*  
**x = points[:, 0]  
y = points[:, 1]**  
*divide el conjunto de datos **points** en dos arreglos x e y. x contendrá todas las primeras columnas de los puntos (las características independientes) y y contendrá todas las segundas columnas de los puntos (las etiquetas o valores reales que queremos predecir).*
**total_cost = np.sum((y - (m * x + b)) ** 2)**  
*Error Cuadrático Medio (MSE). Mide el promedio de los errores al cuadrado entre las predicciones del modelo y los valores reales. Cuanto menor sea el MSE, mejor será el ajuste del modelo a los datos.*  
*np.sum((y - (m * x + b)) ** 2) Esta expresión primero calcula el error cuadrático para cada punto (y - (m * x + b)) ** 2 y luego suma estos errores cuadráticos con np.sum(). Esto da como resultado el **MSE total** para todo el conjunto de datos.*  
Entonces, en esta línea, se está calculando el MSE entre las predicciones (m * x + b) y los valores reales y, y se está sumando sobre todos los puntos del conjunto de datos. Esto da como resultado el costo total asociado con los parámetros b y m.
**return total_cost / len(points)**  
*Devuelve el costo promedio, que es el costo total dividido por el número de puntos. Esto proporciona una medida del error promedio del modelo.*

>>>/CARBON-code
def run_gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations):
    # Initialize the intercept and slope parameters
    b = initial_b
    m = initial_m
    # List to store the history of costs in each iteration
    cost_history = []

    # Loop iterating over the specified number of iterations. For each iteration, optimize b, m and compute its cost
    for i in range(num_iterations):
        # Calculate the current cost and append it to the history
        cost_history.append(calculate_cost(b, m, points))
        
        # Update the parameters using the update function
        b, m = update_parameters(b, m, np.array(points), learning_rate)

    # Return the final values of b and m, along with the cost history
    return [b, m, cost_history]

def calculate_gradients(b_current, m_current, points, learning_rate):
    # Initialize gradients to zero
    m_gradient = 0
    b_gradient = 0
    N = float(len(points))

    # Calculate Gradients
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        m_gradient += - (2/N) * x * (y - (m_current * x + b_current))
        b_gradient += - (2/N) * (y - (m_current * x + b_current))
    
    # Update current m and b using gradients and learning rate
    m_updated = m_current - learning_rate * m_gradient
    b_updated = b_current - learning_rate * b_gradient

    # Return updated parameters
    return b_updated, m_updated
>>>/CARBON-code

Implementa el algoritmo de descenso de gradiente (Gradient Descent) para encontrar los parámetros óptimos b y m de la regresión lineal. Gradient Descent es el algoritmo de optimización utilizado para minimizar la función de costo. Básicamente, busca el mínimo de la función moviéndose en la dirección del gradiente negativo. Esto implica ajustar los parámetros del modelo (pendiente y término independiente) iterativamente. Se define una función para calcular el gradiente (derivadas parciales) de la función de costo respecto a los parámetros, y otra función para actualizar los parámetros con el gradiente y una tasa de aprendizaje (learning rate).

"Optimizar" significa encontrar los valores de los parámetros (en este caso b y m) que minimizan una función de costo específica. El objetivo es ajustar el modelo para que se ajuste lo mejor posible a los datos de entrenamiento.

#### def run_gradient_descent 
*Esta función realiza el proceso de descenso de gradiente, actualizando los parámetros b y m en cada iteración y registrando el historial de costos. Al final, devuelve los valores finales de b y m junto con el historial de costos.*

**b = initial_b  
m = initial_m**  
*Se inicializan los valores de b y m con los valores iniciales proporcionados por el usuario.*
**cost_history = []**  
*Se crea una lista cost_history para almacenar el historial de costos en cada iteración.*
**for i in range(num_iterations)**  
*Se inicia un bucle que itera sobre el número de iteraciones especificadas.*  
**cost_history.append(calculate_cost(b, m, points))**  
*Se calcula el costo actual utilizando la función calculate_cost con los valores actuales de b y m, y se agrega a cost_history.*  
**b, m = update_parameters(b, m, np.array(points), learning_rate)**  
*Se actualizan los valores de b y m utilizando la función update_parameters.*  
**return [b, m, cost_history]**  
*Se devuelven los valores finales de b y m, junto con el historial de costos.*  

#### def calculate_gradients 
*Esta función calcula los gradientes que guían el proceso de optimización.*

**m_gradient = 0  
b_gradient = 0  
N = float(len(points))**  
*Se inicializan los gradientes para m y b en cero. N se establece como el número total de puntos de datos.*  
**for i in range(len(points))**  
*Este bucle itera sobre cada punto de datos. x e y son las coordenadas del punto de datos actual.*  
**m_gradient += - (2/N) * x * (y - (m_current * x + b_current))  
b_gradient += - (2/N) * (y - (m_current * x + b_current))**  
*Se calculan las derivadas parciales de la función de costo con respecto a m y b. Estas derivadas se acumulan en m_gradient y b_gradient.*  
**m_updated = m_current - learning_rate * m_gradient  
b_updated = b_current - learning_rate * b_gradient**  
*Se usan los gradientes y el learning rate para actualizar los valores actuales de m y b.*  
**return b_updated, m_updated**  
*Se devuelven los valores actualizados de b y m se devuelven como una tupla.*

**Una vez definidas las funciones, ejecutaremos el proceso de descenso de gradiente para obtener los parámetros optimizados b y m que minimizan la función de costo. Luego imprimiremos estos valores optimizados y el error asociado.**

>>>/CARBON-code
#Running run_gradient_descent() to get optimized parameters b and m
b, m, cost_graph = run_gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations)

#Print optimized parameters
print('Optimized b:', b)
print('Optimized m:', m)

#Print error with optimized parameters
minimized_cost = compute_cost(b, m, points)
print('Minimized cost:', minimized_cost)
>>>/CARBON-code

**b, m, cost_graph = run_gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations)**  
*Llamamos a la función **run_gradient_descent** con los argumentos especificados (points, initial_b, initial_m, learning_rate y num_iterations). Devuelve los valores optimizados de b y m, así como una lista de valores de costo a lo largo de las iteraciones que se almacena en cost_graph.*  
*"b, m, cost_graph =" Desempaquetado de tuplas. Cuando una función devuelve múltiples valores, pueden ser asignados directamente a múltiples variables en una sola línea de código. Esto significa que b tomará el primer valor que la función devuelve, m tomará el segundo valor y cost_graph tomará el tercero.*  
**print('Optimized b:', b)  
print('Optimized m:', m)**  
*Imprime los valores optimizados de b y m*  
**minimized_cost = compute_cost(b, m, points)  
print('Minimized cost:', minimized_cost)**  
*Variable minimized_cost para almacenar el resultado del cálculo del costo minimizado. Se imprime el costo minimizado (error) asociado con los parámetros optimizados*

***

## ML-LinearRegression-LifeExpectancy

diagrama

### Life_Expectancy_Analysis.ipynb

>>>/CARBON-code
