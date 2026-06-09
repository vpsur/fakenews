# 🔍 Fake News Detector

**Victor Ramirez Muñoz**

---

## El Problema

El problema que he escogido es un detector de noticias falsas. Este proyecto tiene muchas aplicaciones en el día a día, sobre todo en el ámbito digital:

- Plataformas digitales y redes sociales
- Medios de comunicación
- Organismos públicos
- Sistemas de moderación automática
- Herramientas de verificación de información

Durante la realización de este proyecto he utilizado modelos que permiten el procesamiento del lenguaje natural (PLN) y la clasificación supervisada.

---

## Arquitectura de Datos

### Datasets de Entrenamiento
Estas fuentes constituyen el núcleo del aprendizaje del modelo debido a su volumen y balanceo:

> **Kaggle — Fake and Real News Dataset**: Compuesto por 23.502 noticias falsas y 21.417 reales. Proporciona una base equilibrada para el entrenamiento inicial.
> [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

> **Hugging Face — Fake News Detection Dataset**: Complementa al anterior con una estructura similar de clasificación binaria (0 = real, 1 = falso).
> [https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English](https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English)

### Datasets de Validación
> **NewsAPI**: Se empleó para extraer noticias actuales mediante su API. Al no poseer un filtro de veracidad, se utilizó para testear el rendimiento del modelo frente a medios desconocidos en tiempo real.
> [https://newsapi.org/docs/get-started#search](https://newsapi.org/docs/get-started#search)

### Pipeline de Datos

Para combinar los datasets se utilizaron los scripts `pipeline_noticias.py` y `glue_etl_job.py`, apoyándose en AWS (Glue, S3, RDS), MongoDB y Kafka.

```
[Kaggle CSV]          ──► S3 (raw/kaggle/)      ──┐
                                                   ├──► AWS Glue (ETL) ──► RDS MariaDB
[HuggingFace Parquet] ──► S3 (raw/huggingface/) ──┤                       (raw_news_combined)
                                                   │
[NewsAPI]  ──► Kafka ──► MongoDB Atlas ────────────┘
```

---

## Análisis Exploratorio de Datos

Tras la ingesta se realizó un análisis exploratorio que reveló las palabras más frecuentes del dataset (Trump, government, etc.), métricas de longitud de título y contenido, y correlaciones entre variables.

Durante este análisis se detectaron registros vacíos y una correlación entre la longitud del texto y la veracidad de la noticia.

### Limpieza aplicada
- Eliminación de duplicados y columnas innecesarias
- Normalización de formatos de fecha → descomposición en año, mes, día y día de la semana
- Eliminación de stopwords
- Eliminación de registros con menos de 15 palabras en el contenido

El dataset pasó de **~89.000 registros a ~59.000** tras la limpieza.

---

## Entrenamiento del Modelo

Se probaron cinco modelos combinando **TF-IDF** (13.006 features) con features numéricas:

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9957 | 0.9945 | 0.9995 | 0.9970 | 0.9997 |
| Multinomial NB | 0.9578 | 0.9755 | 0.9650 | 0.9702 | 0.9901 |
| LinearSVC | 0.9964 | 0.9954 | 0.9995 | 0.9975 | 0.9998 |
| MLP | 0.9955 | 0.9944 | 0.9994 | 0.9969 | 0.9997 |
| Keras | 0.9953 | 0.9945 | 0.9989 | 0.9967 | 0.9997 |

Se verificó la ausencia de data leakage — los títulos duplicados correspondían a plantillas genéricas de agencias de noticias con contenido completamente distinto.

---

## Representación Gráfica

Se generaron las siguientes visualizaciones:
- **Curvas de aprendizaje** — todos los modelos convergen correctamente sin overfitting
- **Curvas ROC** — AUC superior a 0.99 en todos los modelos
- **Distribución de predicciones** — modelos muy polarizados excepto LinearSVC que mostraba mayor incertidumbre

---

## Despliegue y Comparación

### Comparativa con modelo de Hugging Face

Se comparó con `hamzab/roberta-fake-news-classification`, entrenado con el mismo dataset:

| Modelo | F1 | ROC-AUC | Tipo |
|---|---|---|---|
| Logistic Regression | 0.9970 | 0.9997 | Propio |
| Multinomial NB | 0.9702 | 0.9901 | Propio |
| LinearSVC | 0.9975 | 0.9998 | Propio |
| MLP | 0.9969 | 0.9997 | Propio |
| Keras | 0.9967 | 0.9997 | Propio |
| RoBERTa (HuggingFace) | 0.9170 | 0.9777 | HuggingFace |

Los modelos propios superan al RoBERTa de HuggingFace en todas las métricas.

### Aplicación Gradio

App desplegada en HuggingFace Spaces usando **Logistic Regression** como modelo final:

- 🌐 **App**: [https://huggingface.co/spaces/vrammun/fake-news-detector](https://huggingface.co/spaces/vrammun/fake-news-detector)
- 🤗 **Modelo**: [https://huggingface.co/vrammun/fake-news-detector](https://huggingface.co/vrammun/fake-news-detector)
- 💻 **Versión local**: `src/app/app_local.py`

---

## Fine-Tuning

Se realizó fine-tuning sobre la **red neuronal Keras** con el dataset `GonzaloA/fake_news` (24.351 artículos adicionales), aplicando la misma limpieza y vectorización que al dataset original.

Se congelaron las primeras capas del modelo y se re-entrenaron únicamente las 4 últimas con `learning_rate=0.0001`.

| Momento | F1 | Accuracy | ROC-AUC |
|---|---|---|---|
| Antes | 0.9965 | 0.9950 | 0.9998 |
| Después | 0.9967 | 0.9954 | 0.9998 |

El fine-tuning mantuvo el rendimiento del modelo con una ligera mejora en F1 y Accuracy.

---

## Conclusión Final

Este proyecto ha demostrado que es posible construir un detector de noticias falsas de alta precisión utilizando técnicas clásicas de PLN. La combinación de **TF-IDF con Logistic Regression** ha resultado ser la solución más eficiente — superando incluso a modelos transformer como RoBERTa con una fracción del coste computacional.

El modelo final alcanza un **F1-Score de 0.9970 y un ROC-AUC de 0.9997**, lo que lo convierte en una herramienta prácticamente perfecta para el dominio de noticias políticas en inglés. Sin embargo, es importante señalar sus limitaciones: el modelo está especializado en el estilo lingüístico de las noticias del período 2016-2018 y puede no generalizar igual de bien ante noticias de otros dominios, idiomas o períodos temporales.

El fine-tuning con datos adicionales confirmó la robustez del modelo — los pesos aprendidos son estables y el rendimiento se mantiene prácticamente intacto ante nuevos datos del mismo dominio.

Como trabajo futuro se plantea explorar modelos de lenguaje más modernos como **DistilBERT** o **RoBERTa** entrenados desde cero con datos más recientes y diversos, así como ampliar la aplicación para soportar múltiples idiomas y dominios temáticos.

---

## Repositorio

```
fakenews/
├── datasets/          # Datos locales
├── models/            # Modelos entrenados (.pkl, .keras)
├── notebooks/         # Notebooks de Kaggle
├── scripts/           # Scripts de ingesta y subida
├── src/app/           # Aplicación Gradio
├── .env.example       # Variables de entorno de ejemplo
└── requirements.txt   # Dependencias
```
