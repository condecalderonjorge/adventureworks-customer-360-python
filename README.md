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
```

### 3.4. Selección del número de clusters

Uso de KElbowVisualizer (yellowbrick) con k entre 2 y 10 para identificar el “codo” en la curva de distorsión.

A partir del análisis, se selecciona k = 4 clusters.

### 3.5. Entrenamieto del modelo de clustering

````python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data_cluster)
data_cluster['Cluster'] = clusters
````

### 3.6. Perfilado de los clusters
Se calculan estadísticas descriptivas por cluster (media de TotalAmount, ingresos, hijos, etc.) para interpretar segmentos de negocio.

A partir de los promedios obtenidos, los clusters pueden interpretarse de forma simplificada como:

- **Cluster 2**– “Clientes premium de alto valor”

Mayor importe medio acumulado (TotalAmount más alto).

Niveles de ingresos (YearlyIncome_Num) y educación más elevados.

Todos han comprado bicicleta (BikePurchase = 1).

- **Cluster 1** – “Clientes de alto valor”

Gasto medio elevado, pero por debajo del cluster premium.

Ingresos y educación por encima de la media.

Todos son compradores de bicicleta.

- **Cluster 0** – “Clientes de valor medio”

Gasto moderado.

Todos han comprado bicicleta.

Perfil sociodemográfico intermedio.

- **Cluster 3** – “Clientes de baja implicación / no compradores de bicicleta”

Representan la mayoría de la base (más de la mitad de los clientes).

Importe medio muy bajo.

Solo una pequeña parte ha comprado bicicleta (BikePurchase ≈ 0,13).

Niveles de ingresos y educación algo menores que los clusters de alto valor.


### 3.7. Evaluación del modelo

````python
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(
    data_cluster[seleccion_columnas],
    data_cluster['Cluster']
)
print(f"Puntuación media del coeficiente de Silhouette: {silhouette_avg:.3f}")
````
La puntuación obtenida es aproximadamente 0,75, lo que indica:

Buena separación entre clusters.

Cohesión interna elevada dentro de cada cluster.

Segmentación útil para extraer insights de negocio y diseñar estrategias diferenciadas.

---

## 4. Conclusiones de negocio

A partir del EDA y de la segmentación con k-means se pueden extraer varias conclusiones:

- **La probabilidad de compra de bicicleta y el importe total gastado** están fuertemente relacionados.

- Los clientes con **mayor nivel de ingresos y educación** tienden a concentrarse en los clusters de mayor valor.

- Existe un segmento muy numeroso de clientes con **baja implicación** (cluster 3), donde el gasto es residual y la compra de bicicleta es poco frecuente.

Ejemplos de líneas de acción:

- **Clusters 1 y 2 (alto valor / premium)**

Diseñar programas de fidelización específicos y ofertas cross-sell/upsell (accesorios, componentes premium).

Comunicación personalizada basada en su histórico de compras.

- **Cluster 0 (valor medio)**

Campañas que incentiven un incremento del ticket medio (packs, mantenimiento, upgrades).

-**Cluster 3 (baja implicación)**

Estrategias de activación: campañas de onboarding, descuentos de primera compra, contenidos educativos sobre bicicletas y movilidad.