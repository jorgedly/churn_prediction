# Predicción de Cancelación de Servicios

Proyecto final del curso Introducción a la Ciencia de Datos y Machine Learning.
Predice si un servicio contratado será cancelado usando características del producto, la cuenta y el negocio al momento de la compra.

## Problema

**Tipo:** Clasificación binaria  
**Target:** cancelado / no cancelado  
**Valor de negocio:** Identificar servicios en riesgo de cancelación antes de que ocurra, permitiendo acciones de retención proactiva.

## Estructura del repositorio

```
churn_prediction/
├── data/
│   └── ot.csv                    # Dataset (ver instrucciones abajo)
├── models/
│   ├── best_model.joblib         # Mejor modelo entrenado
│   ├── encoder.joblib            # OrdinalEncoder ajustado
│   └── scaler.joblib             # StandardScaler ajustado
├── src/
│   ├── preprocessing.py          # Limpieza y agrupación de productos
│   ├── features.py               # Ingeniería de características y encoding
│   └── evaluation.py             # Métricas y visualizaciones
├── order_task_churn_prediction.ipynb   # Notebook completo
├── presentacion.ipynb            # Notebook resumido para presentación
├── generate_data.py              # Generador de datos sintéticos
├── predict.py                    # Script de predicción con nuevos datos
├── run.py                        # Script de setup y ejecución completa
├── requirements.txt
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Datos

El dataset contiene registros de servicios contratados con información de producto,
cuenta, tipo de entidad de negocio y canal de adquisición.

Para usar datos propios, coloca el CSV en `data/ot.csv`.
Para usar datos sintéticos (sin datos reales):

```bash
python generate_data.py --n 5000 --output data/ot.csv
```

## Ejecución rápida (recomendado)

```bash
# Con datos reales (coloca tu CSV en data/ot.csv primero)
python run.py

# Con datos sintéticos generados automáticamente
python run.py --generar

# Generar 10,000 filas sintéticas
python run.py --generar --n 10000

# Notebook corto de presentación
python run.py --notebook presentacion

# Abrir MLflow UI al terminar
python run.py --mlflow
```

## Ejecución manual

```bash
jupyter notebook order_task_churn_prediction.ipynb
```

Ejecutar todas las celdas en orden. Al finalizar, los modelos se guardan en `models/`.

## Ver experimentos en MLflow

```bash
mlflow ui
# Abrir http://localhost:5000
```

## Generar predicciones con nuevos datos

```bash
python predict.py --input data/nuevos_datos.csv --output predicciones.csv
```

## Modelos comparados

| Modelo | Justificación |
|---|---|
| Perceptrón | Baseline lineal simple |
| Regresión Logística | Clasificador lineal interpretable |
| Árbol de Decisión | No lineal, no requiere escalamiento |
| Random Forest | Ensemble robusto, provee feature importance |

## Métricas principales

- **F1-macro:** métrica principal (dataset desbalanceado)
- **ROC-AUC:** capacidad discriminativa del modelo
- **Accuracy:** referencia secundaria
