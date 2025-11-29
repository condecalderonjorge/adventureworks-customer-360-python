# Customer 360 & Segmentación de Clientes en Python – AdventureWorks

Proyecto de análisis 360º de clientes para una empresa de retail de bicicletas (AdventureWorks).  
El objetivo es entender el comportamiento de compra y el perfil sociodemográfico de los clientes a partir de un dataset de ~18.000 registros, y construir una **segmentación de clientes basada en *k-means*** para apoyar decisiones de marketing y CRM.

El análisis se ha desarrollado íntegramente en **Python** (pandas, seaborn, scikit-learn), combinando:
- **EDA (Exploratory Data Analysis)** para explorar el comportamiento de los clientes.
- **Preparación de datos** (limpieza, codificación de variables categóricas, detección de outliers).
- **Modelado no supervisado** (clustering con *k-means*).
- **Evaluación de la segmentación** con métrica de **Silhouette**.

---

## 1. Dataset

El dataset original (`data/dataset_AW.csv`) contiene información a nivel de cliente:

- Variables de comportamiento:
  - `TotalAmount`: importe total acumulado de compras.
  - `BikePurchase`: indicador (0/1) de si el cliente ha comprado bicicleta.
- Variables demográficas:
  - `Age`: edad.
  - `TotalChildren`: número de hijos.
  - `MaritalStatus`: estado civil.
  - `Gender`: género.
- Variables socioeconómicas (codificadas):
  - `YearlyIncome_Num`: categoría de nivel de ingresos (0 a 4, de menor a mayor).
  - `Education_Num`: nivel educativo (0 a 4).
  - `Occupation_Num`: categoría de ocupación.
  - `CountryRegionCode_Num`: región de residencia, codificada.

Número de filas: **18.484**  
Número de columnas: **15** (tras la preparación final).

---

## 2. Stack y librerías utilizadas

El proyecto está desarrollado con:

- **Python 3**
- **pandas**, **numpy** – manipulación y preparación de datos.
- **matplotlib**, **seaborn**, **plotly.express** – visualización (histogramas, boxplots, heatmaps, gráficos interactivos).
- **scikit-learn**:
  - `KMeans` – clustering de clientes.
  - `silhouette_score`, `silhouette_samples` – evaluación de la calidad de los clusters.
- **yellowbrick** – búsqueda visual del número óptimo de clusters con el método del codo (*Elbow method*).

---

## 3. Flujo de trabajo

El análisis sigue un flujo clásico de proyecto de datos:

### 3.1. Entendimiento de los datos (EDA)

Dentro del notebook `notebooks/adventureworks_customer_360.ipynb` se realiza:

- **Exploración inicial**:
  - `data.info()` y `data.describe()` para entender tipos y rangos.
  - Revisión de valores faltantes y consistencia de las variables.
- **Distribuciones**:
  - Histogramas de variables numéricas (`TotalAmount`, `Age`, `TotalChildren`, etc.).
- **Detección de outliers**:
  - Boxplots globales y específicos para `TotalAmount`.
- **Correlaciones**:
  - Matriz de correlación y heatmap entre variables numéricas:
    - `TotalAmount` muestra una **alta correlación positiva** con `BikePurchase`.
    - Relación positiva moderada con `YearlyIncome_Num` y `Education_Num`.
    - Correlación negativa ligera con `TotalChildren` y `Age`.

Este EDA sirve como base para entender qué variables tienen más potencial para explicar el valor del cliente.

### 3.2. Preparación de datos

Se realizan varias transformaciones:

- Conversión de variables categóricas a versión numérica:
  - `Gender` → `Gender_Num` (por ejemplo, F = 0, M = 1).
  - `MaritalStatus` → `MaritalStatus_Num` (casado/soltero codificado).
- Revisión de valores nulos y tratamiento de inconsistencias.
- Selección de las variables más relevantes para la segmentación de clientes.

### 3.3. Modelado: segmentación con *k-means*

Para la segmentación se construye un subconjunto de variables relevantes:

```python
seleccion_columnas = [
    'BikePurchase',
    'TotalAmount',
    'TotalChildren',
    'Education_Num',
    'Occupation_Num',
    'YearlyIncome_Num'
]

data_cluster = data[seleccion_columnas]
