<p align="center"><img src="" /></p>

## <p align="center">ML-LinearRegression-StudentStudyHours</p>

*Realizaremos un análisis de regresión lineal simple para encontrar la mejor recta que se ajusta a los datos y luego utilizar esa recta para hacer predicciones sobre las calificaciones de los estudiantes.*

Vamos a implementar dos métodos (uno manual y otro mas automatizado) para calcular la regresión lineal y poder predecir calificaciones en base a un número de horas de estudio. Además utilizaremos distintas formas de cargar, analizar y representar los datos:

#### Caso 1: Student_Study_Hours_Analysis_1.ipynb  
**Carga y Exploración de Datos**:  
Libreria Numpy.  
**Implementación de la Regresión Lineal**:  
Implementaremos manualmente el algoritmo de descenso de gradiente y otros cálculos para encontrar los parámetros de la regresión lineal.  
**Visualizaciones**:  
Liberia matplotlib.  
**Predicciones**:  
Calcula manualmente las predicciones utilizando los parámetros de la regresión lineal encontrados.

![diagram_caso1copia](https://github.com/ccalvop/ML-LinearRegression/assets/126183973/01945bec-8457-47f7-a2c9-139e3a5982bc)

#### Caso 2: Student_Study_Hours_Analysis_2.ipynb  
**Carga y Exploración de Datos**:  
Liberias Numpy y Pandas.  
**Implementación de la Regresión Lineal**:  
Liberia Scikit-learn para crear y entrenar directamente un modelo de regresión lineal.  
**Visualizaciones**:  
Libreria matplotlib y seaborn para una mayor variedad de visualizaciones.  
**Predicciones**:  
Utilizaremos el modelo de regresión lineal entrenado con sklearn para hacer predicciones.

![diagram_caso2 copia](https://github.com/ccalvop/ML-LinearRegression/assets/126183973/4b45fdea-84dc-4f4d-83f6-cf0b5a87479e)
    
En ambos casos, el resultado final es el mismo: se obtiene **una línea de regresión** que mejor se ajusta a los datos, y se pueden realizar predicciones utilizando esta línea. La solución en el caso 1 se centra en la comprensión y la implementación detallada del algoritmo mientras que el caso 2 se centra en la utilización de sklearn, que proporciona una interfaz más rápida y sencilla para implementar la regresión lineal. Compararemos ambos resultados.

***

## Student_Study_Hours_Analysis_1.ipynb

En una regresión lineal tratamos de encontrar una recta (representada por la ecuación y = m x + b) que mejor se ajuste a los datos proporcionados. El objetivo matemático es encontrar los valores de m y b que hace las predicciones más cercanas a los datos reales. En el notebook 1 

```python
#imports
import numpy as np
import matplotlib.pyplot as plt

#magic command
%matplotlib inline
```

**import numpy as np**   
NumPy es una biblioteca para realizar operaciones matriciales y numéricas en Python. Importamos la librería NumPy con el alias np para abreviar. En lugar de escribir numpy cada vez que necesitas usar una función de esta librería, escribimos np. Por ejemplo, en lugar de numpy.array(), puedes escribir np.array().

**import matplotlib.pyplot as plt**  
Matplotlib es una biblioteca gráfica que se utiliza para crear visualizaciones, como gráficos y gráficos. matplotlib.pyplot es un módulo dentro de Matplotlib que proporciona una interfaz similar a la de MATLAB para crear gráficos. La convención de renombrar este módulo como plt es común y facilita el uso.

**%matplotlib inline**  
Es una "comando mágico" en Jupyter Notebook que permite que las gráficas generadas con Matplotlib se muestren directamente en la salida del cuaderno, en lugar de abrirse en una ventana emergente separada

```python
#Load data
points = np.genfromtxt('score.csv', delimiter=',', skip_header=1)

#Extract columns
x = np.array(points[:,0])
y = np.array(points[:,1])

#Plot the dataset
plt.scatter(x,y)
plt.xlabel('Horas de estudio diarias')
plt.ylabel('Resultado del examen')
plt.title('Dataset')
plt.show()
```

#Load data  
**points = np.genfromtxt('score.csv', delimiter=',', skip_header=1)**  
Este código utiliza la función genfromtxt de la biblioteca NumPy para cargar datos desde un archivo CSV llamado 'score.csv'. El parámetro **delimiter=','** indica que las columnas en el archivo CSV están separadas por comas. El parámetro **skip_header=1** indica que se debe omitir la primera fila del archivo CSV al cargar los datos. En nuestro caso la primera fila es un encabezado con etiquetas, y estamos indicando que no queremos cargarlo como datos. Los datos se cargan en la variable points, que será una matriz NumPy que contiene los datos del archivo CSV.

#Extract columns  
**x = array(points[:,0])**  
Se extrae la primera columna de la matriz points y se almacena en la variable x. 
(La expresión [:,0] se utiliza para seleccionar elementos de una matriz NumPy en función de su posición. En este caso selecciona todas las filas de la matriz y solo la primera columna)  
**y = array(points[:,1])**  
Se extrae la segunda columna de la matriz points y se almacena en la variable y

#Plot the dataset   
**plt.scatter(x,y)**
*Utiliza la función scatter de Matplotlib para crear un gráfico de dispersión de los datos. Se pasa x como los valores en el eje X (horas de estudio) y y como los valores en el eje Y (puntuaciones de los exámenes). Esto crea un gráfico que muestra cómo las puntuaciones de los exámenes están relacionadas con las horas de estudio.*  
**plt.xlabel('Horas de estudio diarias')**  
**plt.ylabel('Resultado del examen')**  
*Estos comandos agregan una etiqueta al eje X y al eje Y del gráfico, indicando que las unidades en el eje X son "Horas de estudio diarias" y en el eje Y son "Resultado del examen".*  
**plt.title('Dataset')**  
*Agrega un título al gráfico.*  
**plt.show()**  
*Esta función muestra el gráfico generadolo en la pantalla.*

...  
img-grafica

```python
#Hyperparameters
learning_rate = 0.01
initial_b = 0
initial_m = 0
num_iterations = 50
```

Los **hiperparámetros** son configuraciones que afectan el proceso de entrenamiento del modelo. Los hiperparámetros específicos de la regresión lineal simple son:  
**learning_rate**: La tasa de aprendizaje determina qué tan grande es el paso que se toma en la dirección del gradiente.  
**initial_b e initial_m**: Son los valores iniciales para los coeficientes de la regresión.  
**num_iterations:** Indica el número de iteraciones de entrenamiento.

*En el repositorio **()** podemos ver otro ejemplo de aprendizaje supervisado: XGBoost. Los hiperparámetros que se han configurado en el ejemplo de XGBoost son específicos de ese algoritmo. Cada algoritmo tiene sus propios hiperparámetros, y estos pueden tener nombres diferentes y servir para propósitos distintos.*  
*(XGBoost es un algoritmo de aprendizaje automático supervisado que se utiliza para tareas de clasificación y regresión. El nombre "XGBoost" proviene de "eXtreme Gradient Boosting". Es una implementación eficiente y escalable de árboles de decisión potenciados por gradiente. Este algoritmo tiene sus propios hiperparámetros que son relevantes para su funcionamiento óptimo. max_depth: Controla la profundidad máxima de los árboles. eta: Es el learning rate, que determina cuánto se ajustan los pesos del modelo en cada iteración. gamma: Parámetro que controla la ganancia mínima necesaria para hacer una partición adicional en un nodo hoja del árbol. min_child_weight: Es el mínimo peso necesario de una hoja del árbol. subsample: Controla la fracción de las instancias que se utilizarán para entrenar cada árbol. objective: Define el objetivo del aprendizaje (en este caso, binary:logistic indica una tarea de clasificación binaria). num_round: Indica el número de iteraciones de entrenamiento.)*

#Hyperparameters  
**learning_rate**  
Este es un hiperparámetro que controla el tamaño de los pasos que el algoritmo de optimización toma durante el entrenamiento. Un valor más bajo hace que los pasos sean más pequeños, lo que puede conducir a una convergencia más precisa, pero también puede hacer que el entrenamiento sea más lento. Un valor demasiado alto puede hacer que el entrenamiento diverja o salte sobre el mínimo óptimo.  
*Un valor de 0.01 es bajo y apropiado para comenzar. Evita oscilaciones excesivas en el proceso de entrenamiento.*  
**initial_b**  
Este es el valor inicial del término independiente (también conocido como sesgo o intercept). En una regresión lineal, es el valor de y cuando x es 0. Al inicio del entrenamiento, el modelo utiliza este valor como punto de partida. 
*Un valor de 0 representa la estimación inicial de la variable dependiente cuando todas las variables independientes son cero.*  
**initial_m**  
Este es el valor inicial de la pendiente de la línea. Representa la tasa de cambio de la variable dependiente respecto a la independiente. Al igual que initial_b, este valor se utiliza como punto de partida en el entrenamiento.  
*Un valor de 0. Inicializarlo en 0 significa que al principio no hay pendiente. No hay un cambio inicial en la variable dependiente en relación con las variables independientes.*  
**num_iterations** Este hiperparámetro determina cuántas veces el algoritmo de entrenamiento recorre todo el conjunto de datos de entrenamiento para ajustar los parámetros del modelo. Cada pasada a través del conjunto de datos se conoce como una iteración. Un número mayor de iteraciones puede permitir que el modelo converja mejor, pero también puede aumentar el tiempo de entrenamiento.  
*Un valor de 50. El modelo debería converger relativamente rápido. Es importante monitorear la curva de aprendizaje para asegurarse de que el modelo no esté sobreajustando o subajustando los datos.*  

```python
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
```

La función **compute_cost** tiene como propósito calcular el costo o la pérdida del modelo en función de los parámetros b (intercepción) y m (pendiente). El objetivo del entrenamiento es encontrar los valores de b y m que minimizan este costo. La función de costo (o función de pérdida) en el contexto de la regresión lineal es una medida de cuánto se desvían las predicciones del modelo de los valores reales. En una regresión lineal, la función de costo comúnmente utilizada es el error cuadrático medio (MSE, por sus siglas en inglés). El objetivo del entrenamiento es minimizar esta función.

**En términos prácticos, estamos buscando la línea de regresión que hace las predicciones más cercanas a los datos reales. Esto se logra ajustando los parámetros m y b de manera iterativa utilizando técnicas como el descenso de gradiente.**

Los parámetros **b** y **m**:  
**b** es el término de intercepción. Representa el valor que la variable dependiente (y en este caso, que serían las calificaciones) tiene cuando la variable independiente (x en este caso, que serían las horas de estudio) es igual a cero. En la mayoría de los casos, este valor no tiene un significado real, pero es importante para la forma de la recta.  
**m** es el coeficiente de la pendiente. Indica cuánto cambia la variable dependiente cuando la variable independiente cambia en una unidad. Por ejemplo, si m es 2, significa que por cada unidad de cambio en x, y aumenta en 2 unidades.

**def compute_cost(b, m, points)**  
*Toma tres argumentos: b, m, y points.     
b y m son los parámetros del modelo que estamos tratando de aprender. Estos parámetros determinan la recta que mejor se ajusta a los datos.  
Points es el conjunto de datos sobre el cual estamos calculando el costo.*  
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

```python
def run_gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations):
    # Initialize the intercept and slope parameters
    b = initial_b
    m = initial_m
    # List to store the history of costs in each iteration
    cost_history = []

    # Loop iterating over the specified number of iterations. For each iteration, optimize b, m and compute its cost
    for i in range(num_iterations):
        # Calculate the current cost and append it to the history
        cost_history.append(compute_cost(b, m, points))
        
        # Update the parameters using the calculate_gradients function
        b, m = calculate_gradients(b, m, np.array(points), learning_rate)

        # Printing values
        print(f'Iteration {i+1}: b = {b}, m = {m}, Cost = {cost_history[-1]}')

    # Return the final values of b and m, along with the cost history
    return [b, m, cost_history]

def calculate_gradients(b_current, m_current, points, learning_rate):
    # Initialize gradients to zero
    m_gradient = 0
    b_gradient = 0
    N = float(len(points))

    # Calculate Gradients
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        m_gradient += - (2/N) * x * (y - (m_current * x + b_current))
        b_gradient += - (2/N) * (y - (m_current * x + b_current))
    
    # Update current m and b using gradients and learning rate
    m_updated = m_current - learning_rate * m_gradient
    b_updated = b_current - learning_rate * b_gradient

    # Return updated parameters
    return b_updated, m_updated
```

Aquí implementamos el algoritmo de descenso de gradiente (Gradient Descent) para encontrar los parámetros óptimos b y m de la regresión lineal. Gradient Descent es el algoritmo de optimización utilizado para minimizar la función de costo. Básicamente, busca el mínimo de la función moviéndose en la dirección del gradiente negativo. Esto implica ajustar los parámetros del modelo (pendiente y término independiente) iterativamente. Se define una función para calcular el gradiente (derivadas parciales) de la función de costo respecto a los parámetros, y otra función para actualizar los parámetros con el gradiente y una tasa de aprendizaje (learning rate).

*"Optimizar"* significa encontrar los valores de los parámetros (en este caso b y m) que minimizan una función de costo específica. El objetivo es ajustar el modelo para que se ajuste lo mejor posible a los datos de entrenamiento.

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
**b, m = calculate_gradients(b, m, np.array(points), learning_rate)**  
*Se actualizan los valores de b y m utilizando la función update_parameters.*  
**print(f'Iteration {i+1}: b = {b}, m = {m}, Cost = {cost_history[-1]}')**  
*Imprimimos información relevante sobre el progreso del algoritmo de descenso de gradiente en cada iteración.*  
**return [b, m, cost_history]**  
*Se devuelven los valores finales de b y m, junto con el historial de costos.*  

#### def calculate_gradients 
*Esta función calcula los gradientes que guían el proceso de optimización.*

**m_gradient = 0  
b_gradient = 0  
N = float(len(points))**  
*Se inicializan los gradientes para m y b en cero. N se establece como el número total de puntos de datos.*  
**for i in range(0, len(points))**  
*Este bucle itera sobre cada punto de datos del conjunto de entrenamiento. x e y son las coordenadas del punto de datos actual.*  
**m_gradient += - (2/N) * x * (y - (m_current * x + b_current))  
b_gradient += - (2/N) * (y - (m_current * x + b_current))**  
*Se calculan las derivadas parciales de la función de costo con respecto a m y b. Estas derivadas se acumulan en m_gradient y b_gradient.*  
**m_updated = m_current - learning_rate * m_gradient  
b_updated = b_current - learning_rate * b_gradient**  
*Se usan los gradientes y el learning rate para actualizar los valores actuales de m y b.*  
**return b_updated, m_updated**  
*Se devuelven los valores actualizados de b y m se devuelven como una tupla.*

**Una vez definidas las funciones, ejecutaremos el proceso de descenso de gradiente para obtener los parámetros optimizados b y m que minimizan la función de costo. Luego imprimiremos estos valores optimizados y el error asociado.**

```python
#Running run_gradient_descent() to get optimized parameters b and m
b, m, cost_graph = run_gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations)

#Print optimized parameters
print('Optimized b:', b)
print('Optimized m:', m)

#Print error with optimized parameters
minimized_cost = compute_cost(b, m, points)
print('Minimized cost:', minimized_cost)
``` 

**b, m, cost_graph = run_gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations)**  
*Llamamos a la función **run_gradient_descent** con los argumentos especificados (points, initial_b, initial_m, learning_rate y num_iterations). Devuelve los valores optimizados de b y m, así como una lista de valores de costo a lo largo de las iteraciones que se almacena en cost_graph.*  
*"b, m, cost_graph =" Desempaquetado de tuplas. Cuando una función devuelve múltiples valores, pueden ser asignados directamente a múltiples variables en una sola línea de código. Esto significa que b tomará el primer valor que la función devuelve, m tomará el segundo valor y cost_graph tomará el tercero.*  

**print('Optimized b:', b)  
print('Optimized m:', m)**  
*Imprime los valores finales optimizados de b y m*  

**minimized_cost = compute_cost(b, m, points)  
print('Minimized cost:', minimized_cost)**  
*Variable minimized_cost para almacenar el resultado del cálculo del costo minimizado. Se imprime el costo minimizado (error) asociado con los parámetros optimizados*

```python
#Plot Cost per Iteration
plt.plot(cost_graph)
plt.xlabel('Número de iterations')
plt.ylabel('Costo')
plt.title('Costo por Iteración')
plt.show()
```
Con este código creamos un gráfico que muestra cómo cambia la función de costo a medida que avanzan las iteraciones del algoritmo de descenso de gradiente.  
Visualizamos cómo la función de costo **disminuye** a medida que el algoritmo de descenso de gradiente avanza a través de las iteraciones.

...  
img grafica

La curva se aplana a partir de x iteraciones, lo que indica que el algoritmo converge a una solución. Entonces podriamos reducir el número de iteraciones sin afectar significativamente la precisión del modelo. Esto sería una optimización importante, ya que menos iteraciones significan un tiempo de entrenamiento más corto.  
(*)Probaremos diferentes hiperparámetros y sus resultados después de visualizar la regresión final obtenida.

### Ahora visualizaremos los resultados del modelo de regresión lineal una vez que ha sido entrenado.

```python
#Plot dataset
plt.scatter(x, y)
#Predict y values
pred = m * x + b
#Plot predictions as line of best fit
plt.title('Modelo de Regresión Lineal: Estimación de Calificaciones')
plt.xlabel('Horas de estudio')
plt.ylabel('Resultados de examenes')
plt.plot(x, pred, c='r')
plt.show()
```

**plt.scatter(x, y)**  
*Utiliza la función scatter de Matplotlib para crear un gráfico de dispersión de los datos*  
**pred (y) = m * x + b**  
*Se calculan las predicciones del modelo utilizando los parámetros optimizados m y b. Representa la mejor línea de ajuste que el modelo ha aprendido.  
Una vez que tenemos los valores optimizados de "m" y "b", podemos usarlos para predecir "y" (calificación) para cualquier valor de "x" (horas de estudio).*  
**plt.title('Modelo de Regresión Lineal: Estimación de Calificaciones')**  
*Se establece el título del gráfico.*  
**plt.xlabel('Horas de estudio') y plt.ylabel('Resultados de examenes')**  
*Estas líneas agregan etiquetas a los ejes x e y, respectivamente.*  
**plt.plot(x, pred, c='r')**  
*Traza la línea de mejor ajuste en el gráfico "pred". La c='r' especifica que el color de la línea será rojo ('r' red en matplotlib).  
**plt.show()**  
*Esta función muestra el gráfico.*  

#### Diferentes hiperparámetros y sus resultados

...  
img

...  
img

...  
img
