# Modelos Autoregresivos Profundos: Fundamentos y Evolución

Resumen de las clases de la asignatura.

## Resumen Ejecutivo

Este documento técnico sintetiza los conceptos fundamentales y los avances recientes en los Modelos Autoregresivos (AR) Profundos, basándose en el análisis de Pablo Martínez Olmos (UC3M). El objetivo central de estos modelos es el modelado probabilístico generativo, permitiendo la estimación de distribuciones de datos complejas mediante la descomposición de la probabilidad conjunta en productos de probabilidades condicionales.

Los puntos clave identificados incluyen:

- **Fundamentos de Modelado:** La transición de distribuciones desconocidas $p(y)$ a modelos parametrizados $p_\theta(y)$ mediante la minimización de funciones de pérdida.
- **MADE (Masked Autoencoders for Distribution Estimation):** La implementación de la propiedad autoregresiva en arquitecturas feedforward mediante el uso de máscaras en las capas ocultas.
- **Modelos Agnósticos al Orden (OA-ARM):** Una evolución que permite generar variables en órdenes aleatorios, superando las limitaciones de los órdenes fijos.
- **OA-ARDM (Order-Agnostic Autoregressive Diffusion Models):** Una técnica avanzada que marginaliza todas las permutaciones posibles y permite predecir múltiples variables enmascaradas simultáneamente.
- **Aplicaciones en Series Temporales:** El uso de redes recurrentes (RNN, LSTM), Transformers y Modelos de Espacio de Estados (SSM) para amortizar la historia de los datos y mejorar el rendimiento multiescala.

---

## 1. Modelado Probabilístico Generativo

El objetivo primordial del modelado generativo es aproximar una distribución de datos real pero desconocida, $p(y)$, utilizando un modelo parametrizado $p_\theta(y)$.

### Proceso de Estimación

1. **Muestras:** Se parte de un conjunto de muestras $y_n$ para $n = 1, \dots, N$.
2. **Optimización:** Se busca el parámetro óptimo $\theta^*$ que minimice la función de pérdida $L_\theta(y_1, \dots, y_N)$, lo cual generalmente equivale a maximizar la verosimilitud de los datos observados.

---

## 2. Redes Bayesianas Profundas y Autorregresión

La autorregresión se basa en la descomposición de la distribución conjunta mediante la regla de la cadena, representada a menudo a través de Gráficos Acíclicos Dirigidos (DAG).

- **Dependencias Probabilísticas:** En un DAG, los nodos representan variables aleatorias y los bordes dirigidos indican dependencias condicionales (un borde de $a$ hacia $b$ indica que $b$ depende condicionalmente de $a$).
- **Estructura AR:** Para cualquier variable aleatoria y cualquier DAG, la distribución se puede expresar como:

$$p_\theta(x) = \prod_{d=1}^{D} p_\theta(x_d \mid x_{<d})$$

Donde cada dimensión $d$ depende únicamente de las dimensiones precedentes.

---

## 3. MADE: Autoencoders con Máscara para Estimación de Distribuciones

MADE es una arquitectura que adapta los autoencoders tradicionales para que respeten la propiedad autoregresiva.

### Mecanismo de Funcionamiento

- **Máscaras de Pesos:** Se introducen máscaras binarias ($M^W$, $M^V$) en las matrices de pesos para asegurar que la unidad de salida $d$ solo dependa de las entradas $x_{<d}$.
- **Asignación de Unidades:** A cada unidad en la capa oculta se le asigna un número entero entre $1$ y $D-1$. Se colocan ceros en la máscara según la regla:

$$M^l_{k',k} = 1 \quad \text{si} \quad m^l(k') \geq m^{l-1}(k)$$

- **Entrenamiento y Muestreo:** Durante el entrenamiento, funciona como una arquitectura feedforward eficiente. Sin embargo, el muestreo sigue siendo secuencial.

### Limitaciones de MADE

- El DAG debe fijarse de antemano.
- El manejo de entradas faltantes puede resultar problemático.
- El cálculo de probabilidades marginales como $p(x_1, x_2)$ es computacionalmente difícil ya que requiere integrar sobre las variables restantes.

---

## 4. Modelos Autoregresivos Agnósticos al Orden (OA-ARM)

Introducidos en foros como ICML 2014 y NeurIPS 2019, estos modelos proponen que el orden de generación de las variables sea una variable latente aleatoria $\sigma$.

- **Principio:** En lugar de un orden fijo, el modelo se entrena para predecir variables en cualquier orden aleatorio extraído de una distribución uniforme sobre todas las permutaciones posibles $S_D$.
- **Objetivo de Aprendizaje:** El logaritmo de la probabilidad se estima promediando sobre los diferentes tiempos $t$ y órdenes $\sigma$.

---

## 5. Modelos de Difusión Autoregresivos Agnósticos al Orden (OA-ARDM)

Esta es una extensión sofisticada que optimiza la eficiencia del aprendizaje y el muestreo.

### Innovaciones Clave

- **Marginalización de Permutaciones:** Se promedia el objetivo a través de todas las dimensiones no observadas en un momento dado $t$.
- **Predicción Múltiple:** En el tiempo $t$, existen exactamente $D - t + 1$ variables enmascaradas. El modelo intenta predecirlas todas simultáneamente.
- **Red Compartida:** Se utiliza una única red neuronal que se comparte para diferentes órdenes $\sigma$ y pasos temporales $t$, utilizando máscaras binarias.

### Algoritmos Principales

| Algoritmo | Descripción Breve |
|---|---|
| **Muestreo en OA-ARDM** | Inicializa $x = 0$, elige un orden $\sigma$ y para cada paso $t$, genera una muestra condicional basada en las variables ya observadas. |
| **Optimización (ELBO)** | Selecciona $t$ y $\sigma$ al azar, calcula una máscara $m$ y computa la pérdida basada en la predicción de las variables restantes. |

---

## 6. Modelos AR Profundos para Series Temporales

En el contexto de series temporales, los datos se presentan como una secuencia $x = [x_0, x_1, \dots, x_T]$.

- **Estructura Temporal:** La distribución se descompone como:

$$p_\omega(x) = \prod p_\omega(x_i \mid x_1, \dots, x_{i-1})$$

- **Amortización de la Historia:** Para manejar dependencias de largo plazo, se utilizan arquitecturas avanzadas:
  - Redes Neuronales Recurrentes (RNN) y LSTM.
  - Transformers (Decoders): utilizan mecanismos de atención para procesar la historia de la serie de manera eficiente.
  - Modelos de Espacio de Estados (SSM).
- **Rendimiento Multiescala:** La eficacia de estos modelos mejora significativamente al analizar los datos en diferentes escalas temporales simultáneamente.

---

## 7. Aplicaciones de Vanguardia

El material analizado destaca implementaciones modernas que utilizan estos principios autoregresivos y de difusión:

- **VideoPoet:** Un modelo de lenguaje de gran escala (LLM) diseñado para la generación de video, capaz de integrar texto, imágenes, flujo óptico y audio. Utiliza decodificadores MAGVIT-v2 y SoundStream para generar salidas de video y audio de forma autoregresiva.
- **Modelos de Difusión en Espacio Latente:** Utilizan un proceso de difusión hacia adelante (añadiendo ruido gradualmente hasta $x_T$) y un proceso de difusión inversa (denoising) mediante una U-Net con mecanismos de atención cruzada (cross-attention) para reconstruir la imagen a partir del ruido, condicionada por mapas semánticos, texto o representaciones de imágenes.
- **XLNet:** Un ejemplo de preentrenamiento autoregresivo generalizado para la comprensión del lenguaje, que utiliza la metodología agnóstica al orden.
