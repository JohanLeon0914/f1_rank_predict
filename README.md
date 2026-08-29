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

En Railway el servicio usa automaticamente la variable `PORT` provista por la
plataforma mediante `railway.json`.

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

## Ejecutar con Docker

Construye y ejecuta la API localmente desde la raíz del proyecto:

```bash
docker build -t f1-ranker-api:local .
docker run --rm -p 8080:8080 f1-ranker-api:local
```

Verifica que el modelo esté disponible en `http://localhost:8080/health`.

## Desplegar la API en Google Cloud Run

Este flujo usa Cloud Build para construir la imagen sin necesitar Docker local y
Artifact Registry para almacenarla.

1. Instala Google Cloud CLI, inicia sesión y selecciona el proyecto:

```bash
gcloud auth login
gcloud auth configure-docker REGION-docker.pkg.dev
gcloud config set project PROJECT_ID
```

2. Define variables en tu terminal, reemplazando los valores de ejemplo:

```bash
PROJECT_ID="mi-proyecto"
REGION="us-central1"
REPOSITORY="f1-images"
SERVICE="f1-ranker-api"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE}:v1"
```

En PowerShell, usa `$env:PROJECT_ID`, `$env:REGION`, etc., o reemplaza los
valores directamente en los comandos siguientes.

3. Habilita las APIs y crea el repositorio de imágenes una sola vez:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create "$REPOSITORY" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Imágenes de la API F1 Ranker"
```

4. Desde la raíz del repositorio, construye y publica la imagen:

```bash
gcloud builds submit --tag "$IMAGE" .
```

5. Despliega el contenedor en Cloud Run. `--allow-unauthenticated` hace que la
API acepte peticiones públicas; omítelo si usarás autenticación IAM:

```bash
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --min 0 \
  --max 3 \
  --allow-unauthenticated
```

6. Obtén la URL y comprueba la API:

```bash
URL=$(gcloud run services describe "$SERVICE" \
  --region "$REGION" \
  --format="value(status.url)")
curl "$URL/health"
curl "$URL/metrics"
```

Para predecir, envía un `POST` a `$URL/predict` con el JSON del contrato
mostrado arriba. La primera petición puede tardar un poco mientras Cloud Run
inicia el contenedor. El modelo y los CSV quedan incluidos en la imagen, por lo
que no hace falta montar un disco ni configurar una base de datos.
