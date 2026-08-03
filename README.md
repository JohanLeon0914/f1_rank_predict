# F1 Ranker

Modelo de ranking para predecir el orden esperado de pilotos en una carrera de Formula 1.

## Entrenamiento

Usa el entorno virtual del proyecto:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_ranker.py
```

El entrenamiento guarda estos artefactos en `models/`:

- `xgb_ranker.json`: modelo XGBoost Ranker.
- `feature_artifacts.joblib`: preprocesador y columnas usadas por el modelo.
- `metrics.joblib`: metricas del ultimo entrenamiento.
- `feature_importance.csv`: importancia de variables.

Metricas actuales del ultimo reentrenamiento:

| Split | NDCG | Spearman | Kendall tau | MAE posicion | Top 1 | Top 3 | Top 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.9461 | 0.6484 | 0.5205 | 3.8266 | 0.4770 | 0.6379 | 0.7649 |
| test | 0.9480 | 0.6477 | 0.5185 | 3.4366 | 0.5115 | 0.6686 | 0.7770 |

## Prediccion por script

```bash
PYTHONPATH=src .venv/bin/python scripts/predict_existing_race_example.py
```

El script toma la ultima carrera disponible en los CSV, la trata como si fuera nueva y devuelve un ranking predicho.

## API HTTP

Instala dependencias si el entorno no tiene FastAPI:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Levanta la API:

```bash
.venv/bin/python scripts/run_api.py
```

Si el puerto `8000` esta ocupado:

```bash
.venv/bin/python scripts/run_api.py --port 8010
```

Endpoints:

- `GET /health`: valida que existan los artefactos del modelo.
- `GET /metrics`: devuelve las metricas guardadas del ultimo entrenamiento.
- `POST /predict`: predice el ranking de una carrera.

### Contrato de `POST /predict`

La API necesita una carrera y una lista de pilotos con datos disponibles antes de la carrera:

```json
{
  "race_id": 1180,
  "circuit_id": 11,
  "race_date": "2026-08-02",
  "participants": [
    {
      "driverId": 844,
      "constructorId": 6,
      "grid": 1,
      "qualifying_position": 1,
      "q1": "1:26.572",
      "q2": "1:25.900",
      "q3": "1:25.100"
    },
    {
      "driverId": 1,
      "constructorId": 131,
      "grid": 2,
      "qualifying_position": 2,
      "q1": "1:26.800",
      "q2": "1:26.000",
      "q3": "1:25.300"
    }
  ]
}
```

Campos requeridos por participante:

- `driverId`: ID del piloto.
- `constructorId`: ID del constructor/equipo.
- `grid`: posicion de largada. Puede ser `null` si no existe.
- `qualifying_position`: posicion en clasificacion. Puede ser `null`.
- `q1`, `q2`, `q3`: tiempos en formato `M:SS.mmm`. Pueden ser `null`.

`race_date` es obligatorio cuando `race_id` no existe en `F1/CSVs/races.csv`; para carreras existentes se puede omitir.

Respuesta:

```json
{
  "race_id": 1180,
  "circuit_id": 11,
  "predictions": [
    {
      "predicted_position": 1,
      "driverId": 844,
      "score": 1.76
    }
  ]
}
```
