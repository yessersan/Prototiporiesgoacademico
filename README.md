
# 🎓 Prototipo Predictivo de Riesgo Académico Estudiantil (UDH)

Repositorio del prototipo de aprendizaje automático desarrollado bajo la metodología **CRISP-DM** para la predicción temprana del nivel de riesgo académico estudiantil ($t+1$) en la **Universidad de Huánuco (UDH)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Django REST](https://img.shields.io/badge/Django_REST-Ready-092E20?logo=django&logoColor=white)

---

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline completo de ciencia de datos y clasificación multiclase en **Google Colab** para identificar estudiantes en **Riesgo Bajo (`0`)**, **Riesgo Medio (`1`)** o **Riesgo Alto (`2`)** en el periodo académico posterior ($t+1$), priorizando la sensibilidad (*Recall*) en la detección de casos críticos y exportando el modelo entrenado (`.joblib`) para su consumo mediante una API en **Django REST Framework**.

### Características principales
* **Privacidad y seudonimización:** Generación de identificadores anónimos mediante hash `SHA-256` en cumplimiento con la **Ley N.º 29733** (Ley de Protección de Datos Personales).
* **Preprocesamiento automatizado:** Pipeline con `ColumnTransformer` (`StandardScaler` para variables numéricas y `OneHotEncoder` para variables categóricas).
* **Evaluación comparativa:** Validación cruzada estratificada (*5-Fold Stratified CV*) y prueba final sobre cuatro algoritmos supervisados (**Regresión Logística**, **Árbol de Decisión**, **Bosque Aleatorio** y **Gradient Boosting**).
* **Explicabilidad y despliegue:** Análisis de importancia de variables (Índice de Gini) y serialización del pipeline completo en `modelo_riesgo_academico_udh.joblib`.

---

## 🗂️ Estructura del Repositorio

```text
Prototiporiesgoacademico/
├── notebooks/
│   └── Prototipo_Riesgo_Academico_UDH.ipynb   # Notebook ejecutado en Google Colab
├── model/
│   └── modelo_riesgo_academico_udh.joblib     # Artefacto exportado para Django REST
├── img/                                       # Capturas y gráficas generadas en Colab
│   ├── distribucion_clases_riesgo.png
│   ├── tabla_comparativa_modelos.png
│   ├── matriz_confusion_rf.png
│   └── importancia_variables.png
├── requirements.txt                           # Dependencias del entorno
└── README.md                                  # Documentación del proyecto
```

---

## 📑 Diccionario de Variables del Dataset

El dataset consta de **1,500 registros** y **12 variables** ajustadas a las reglas académicas de la UDH:

| Variable | Tipo | Descripción |
| :--- | :--- | :--- |
| `estudiante_id` | `string` | Identificador seudonimizado mediante `SHA-256` (10 caracteres). |
| `escuela_profesional` | `string` | Escuela académica (*Ingeniería de Sistemas, Derecho, Administración, Medicina Humana, Psicología, Arquitectura*). |
| `ciclo` | `int` | Ciclo académico actual del estudiante (`2` a `10`). |
| `promedio_ponderado_acumulado` | `float` | Promedio ponderado histórico acumulado en escala vigesimal (`0.0` – `20.0`). |
| `promedio_ultimo_periodo` | `float` | Promedio obtenido en el último periodo concluido ($t$). |
| `creditos_matriculados` | `int` | Total de créditos matriculados acumulados. |
| `creditos_aprobados` | `int` | Total de créditos aprobados acumulados. |
| `proporcion_avance` | `float` | Ratio entre créditos aprobados y créditos matriculados (`0.0` – `1.0`). |
| `cursos_desaprobados_acumulados` | `int` | Cantidad histórica de asignaturas desaprobadas. |
| `cursos_repetidos_acumulados` | `int` | Cantidad de asignaturas llevadas en segunda o más matrículas. |
| `tendencia_rendimiento` | `float` | Variación del promedio entre el último y el penúltimo periodo. |
| `nivel_riesgo_t1` | `int` | **Variable objetivo ($t+1$):** `0` = Riesgo Bajo, `1` = Riesgo Medio, `2` = Riesgo Alto. |

---

## ⚙️ Fases del Pipeline (CRISP-DM)

1. **Fase 1 – Configuración del Entorno:** Carga de librerías (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`) y fijación de semilla aleatoria (`seed=42`).
2. **Fase 2 – Generación y Seudonimización de Datos:** Simulación de 1,500 registros estudiantiles con anonimización `SHA-256`.
3. **Fase 3 – Análisis Exploratorio y Partición:** División estratificada **70% entrenamiento / 30% prueba** (`stratify=y`).
4. **Fase 4 – Preprocesamiento y Modelado:** Configuración del `ColumnTransformer` y definición de los 4 clasificadores.
5. **Fase 5 – Validación Cruzada y Métricas:** Evaluación con `StratifiedKFold(n_splits=5)` comparando *Accuracy*, *Precision*, *Recall Macro*, *F1-Score Macro* y *Recall en Riesgo Alto*.
6. **Fase 6 – Matriz de Confusión:** Evaluación detallada del mejor modelo (**Random Forest** con `class_weight='balanced'`).
7. **Fase 7 – Explicabilidad del Modelo:** Ranking de las 10 variables predictoras con mayor impacto según importancia de Gini.
8. **Fase 8 – Exportación del Artefacto:** Empaquetado del pipeline y metadatos en `modelo_riesgo_academico_udh.joblib` para integración con **Django REST Framework**.

---

## 📸 Resultados y Visualizaciones del Notebook (Google Colab)

### 1. Distribución de Clases de Riesgo Académico ($t+1$)
![Distribución de Clases de Riesgo Académico](img/distribucion_clases_riesgo.png)

### 2. Comparativa de Desempeño de Algoritmos
![Tabla Comparativa de Modelos](img/tabla_comparativa_modelos.png)

### 3. Matriz de Confusión – Bosque Aleatorio (Random Forest)
![Matriz de Confusión Random Forest](img/matriz_confusion_rf.png)

### 4. Top 10 Variables Predictoras de Riesgo Académico
![Importancia de Variables](img/importancia_variables.png)

---

## 🚀 Cómo Ejecutar el Proyecto

### En Google Colab
1. Abre el cuaderno `.ipynb` en **Google Colaboratory**.
2. Ejecuta todas las celdas secuencialmente (`Entorno de ejecución > Ejecutar todas`).
3. Descarga el archivo generado `modelo_riesgo_academico_udh.joblib` desde el panel de archivos izquierdo de Colab.

### En Entorno Local
1. Clona este repositorio:
   ```bash
   git clone [https://github.com/yessersan/Prototiporiesgoacademico.git](https://github.com/yessersan/Prototiporiesgoacademico.git)
   cd Prototiporiesgoacademico
   ```
2. Instala las dependencias:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib
   ```
3. Ejecuta el notebook desde Jupyter Notebook o VS Code.

---

## 👥 Autores

* **Yezer Santiago Juan de Dios**
* **Guadalupe Gonzales Pino**
* **Piero Avelino Martin**

*Facultad de Ingeniería – Programa Académico de Ingeniería de Sistemas e Informática, Universidad de Huánuco (2026).*
