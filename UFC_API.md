# API UFC Fight Predictor

Documentacion del endpoint UFC agregado al proyecto. La API sigue corriendo desde el mismo entrypoint existente:

```bash
.venv-linux/bin/python scripts/run_api.py --host 0.0.0.0 --port 8000
```

Base URL local:

```text
http://localhost:8000
```

## Endpoints

### GET `/ufc/health`

Verifica que existan los artefactos minimos del modelo UFC:

- `models/ufc/ufc_winner_xgb.json`
- `models/ufc/ufc_feature_artifacts.joblib`
- `models/ufc/ufc_metrics.joblib`

Respuesta correcta:

```json
{
  "status": "ok"
}
```

Si falta algun artefacto responde `503` con `missing_files`.

### GET `/ufc/metrics`

Devuelve las metricas y el split temporal usados en el ultimo entrenamiento.

Respuesta principal:

```json
{
  "metrics": {
    "validation": {
      "accuracy": 0.5985,
      "roc_auc": 0.6268,
      "log_loss": 0.6632,
      "brier_score": 0.235,
      "precision": 0.6149,
      "recall": 0.7967,
      "f1": 0.6941
    },
    "test": {
      "accuracy": 0.6115,
      "roc_auc": 0.6463,
      "log_loss": 0.6605,
      "brier_score": 0.233,
      "precision": 0.6151,
      "recall": 0.7964,
      "f1": 0.6941
    }
  }
}
```

### POST `/ufc/predict-fight`

Predice el ganador de una pelea desde la perspectiva de dos esquinas: `red` y `blue`.

Request minimo:

```json
{
  "red_fighter_id": "ad32471f01e7b1a5",
  "blue_fighter_id": "79899ecf62020f6d"
}
```

Request completo:

```json
{
  "red_fighter_id": "ad32471f01e7b1a5",
  "blue_fighter_id": "79899ecf62020f6d",
  "fight_date": "2026-09-04",
  "weight_class": "Lightweight",
  "title_fight": false
}
```

Campos:

- `red_fighter_id`: requerido. ID del peleador en `UFC/fighter.csv`.
- `blue_fighter_id`: requerido. ID del rival en `UFC/fighter.csv`.
- `fight_date`: opcional. Fecha `YYYY-MM-DD`. Si no se envia, usa el dia posterior al ultimo evento historico del CSV.
- `weight_class`: opcional. Si no se envia, se infiere desde el historial reciente de ambos peleadores.
- `title_fight`: opcional, por defecto `false`.

Respuesta:

```json
{
  "prediction": {
    "winner_side": "red",
    "winner_fighter_id": "ad32471f01e7b1a5",
    "winner_name": "Jeremy Jackson",
    "red_win_probability": 0.6782,
    "blue_win_probability": 0.3218,
    "confidence_pct": 67.8242,
    "model_output_note": "Probabilities are model estimates, not guarantees or betting advice."
  },
  "fight": {
    "fight_date": "2026-08-09",
    "weight_class": "Lightweight",
    "title_fight": false
  },
  "fighters": {
    "red": {
      "fighter_id": "ad32471f01e7b1a5",
      "name": "Jeremy Jackson",
      "history": {},
      "recent_fights": []
    },
    "blue": {
      "fighter_id": "79899ecf62020f6d",
      "name": "Joe Stevenson",
      "history": {},
      "recent_fights": []
    }
  },
  "head_to_head": {
    "h2h_total_fights": 1,
    "h2h_red_wins": 0,
    "h2h_blue_wins": 1,
    "h2h_red_win_rate": 0,
    "h2h_win_diff": -1
  },
  "analysis": {
    "top_contributions": [],
    "bias": 0.0,
    "global_feature_importance": [],
    "model_metrics": {},
    "explanation_note": "top_contributions moves the raw model log-odds toward red or blue. Historical fight statistics are calculated only from fights before fight_date."
  }
}
```

## Interpretacion

- `red_win_probability` y `blue_win_probability` son probabilidades estimadas por el modelo.
- `confidence_pct` es la mayor de las dos probabilidades multiplicada por 100.
- `top_contributions.direction = "red"` significa que esa feature empuja el log-odds hacia victoria roja.
- `top_contributions.direction = "blue"` significa que esa feature empuja el log-odds hacia victoria azul.
- Las contribuciones explican el modelo, no causalidad real.
- La prediccion no debe usarse como consejo financiero ni de apuestas.

## Errores

`400 Bad Request`:

- Falta el modelo entrenado.
- El `fighter_id` no existe en `UFC/fighter.csv`.
- Los dos IDs son iguales.
- La fecha no se puede parsear.

`422 Unprocessable Entity`:

- JSON invalido.
- Falta `red_fighter_id` o `blue_fighter_id`.

## Entrenar o reentrenar

```bash
.venv-linux/bin/python scripts/train_ufc_model.py
```

El entrenamiento actual genera:

- `models/ufc/ufc_winner_xgb.json`
- `models/ufc/ufc_feature_artifacts.joblib`
- `models/ufc/ufc_metrics.joblib`
- `models/ufc/ufc_feature_importance.csv`
- `reports/ufc/ufc_feature_importance.svg`
- `reports/ufc/ufc_roc_curve.svg`
- `reports/ufc/ufc_metrics.json`
- `reports/ufc/ufc_model_dataset_preview.csv`

