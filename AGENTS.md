# AGENTS.md

Guia para el frontend que consume la API de predicciones F1 Ranker.

## Resumen

Esta app expone una API HTTP FastAPI para predecir el ranking esperado de pilotos en una carrera de Formula 1. El frontend solo necesita consumir tres endpoints:

- `GET /health`: confirma que la API y los artefactos del modelo estan disponibles.
- `GET /metrics`: devuelve metricas del ultimo entrenamiento.
- `POST /predict`: genera la prediccion de ranking y devuelve datos de analisis para explicar el resultado.

La API no expone endpoints de catalogo para pilotos, constructores/equipos, circuitos o calendario. El frontend debe obtener esos datos de otra fuente, mantenerlos localmente o pedirlos a otro backend.

## Base URL

Local:

```text
http://localhost:8000
```

Docker local:

```text
http://localhost:8080
```

Produccion:

```text
Usar la URL publica del despliegue, por ejemplo Cloud Run o Railway.
```

## Requisito de CORS

El backend actual no configura `CORSMiddleware`. Si el frontend corre en otro origen, por ejemplo `http://localhost:3000`, el navegador puede bloquear las peticiones por CORS aunque la API responda correctamente.

Para un frontend web, validar uno de estos enfoques antes de integrar:

- servir frontend y API bajo el mismo origen;
- agregar CORS al backend;
- usar un proxy/server route en el frontend.

## Flujo Recomendado

1. Al iniciar la app, llamar `GET /health`.
2. Si responde `200`, habilitar la pantalla de prediccion.
3. Construir el payload de `POST /predict` con la carrera, circuito y participantes.
4. Mostrar `predictions` ordenado por `predicted_position`.
5. Usar `analysis_summary` y `predictions[].analysis` para construir el dashboard explicativo.
6. Si `POST /predict` devuelve error, mostrar `detail` al usuario o registrarlo para diagnostico.

## GET /health

Verifica que existan los artefactos minimos del modelo:

- `models/xgb_ranker.json`
- `models/feature_artifacts.joblib`
- `models/metrics.joblib`

Respuesta correcta:

```json
{
  "status": "ok"
}
```

Errores esperados:

- `503`: faltan artefactos del modelo.

Ejemplo de error:

```json
{
  "detail": {
    "missing_files": [
      "/path/models/xgb_ranker.json"
    ]
  }
}
```

## GET /metrics

Devuelve las metricas guardadas del ultimo entrenamiento. Usar este endpoint para mostrar calidad historica del modelo, no para hacer predicciones.

Errores esperados:

- `503`: no existe `models/metrics.joblib`.

## POST /predict

Predice el orden final esperado para los pilotos enviados. Tambien devuelve informacion explicativa para dashboards: importancia global de variables, grupos de features por piloto y contribuciones del modelo para cada prediccion.

Endpoint:

```text
POST /predict
Content-Type: application/json
```

### Request

```json
{
  "race_id": 1180,
  "circuit_id": 39,
  "race_date": "2026-08-23",
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

### Campos de la Carrera

`race_id`:

- Tipo: `number` entero.
- Requerido.
- Identificador de la carrera.
- Si no existe en `F1/CSVs/races.csv`, el frontend debe enviar tambien `race_date`.

`circuit_id`:

- Tipo: `number` entero.
- Requerido.
- Identificador del circuito.

`race_date`:

- Tipo: `string | null`.
- Formato recomendado: `YYYY-MM-DD`.
- Requerido cuando `race_id` no existe en los CSV historicos del backend.
- Puede omitirse o ser `null` para carreras existentes en `F1/CSVs/races.csv`.
- La fecha define que historico puede usar el modelo: solo carreras anteriores a esa fecha.

`participants`:

- Tipo: array.
- Requerido.
- Minimo: 2 participantes.
- Debe incluir a todos los pilotos que se quieren ordenar en la prediccion.

### Campos por Participante

`driverId`:

- Tipo: `number` entero.
- Requerido.
- ID del piloto.

`constructorId`:

- Tipo: `number` entero.
- Requerido.
- ID del constructor/equipo.

`grid`:

- Tipo: `number | null`.
- Requerido como clave, puede ser `null`.
- Posicion de largada.

`qualifying_position`:

- Tipo: `number | null`.
- Requerido como clave, puede ser `null`.
- Posicion en clasificacion.

`q1`, `q2`, `q3`:

- Tipo: `string | number | null`.
- Requeridos como claves, pueden ser `null`.
- Formato recomendado si hay tiempo: `M:SS.mmm`, por ejemplo `1:26.572`.
- Usar `null` si el piloto no registro tiempo en esa sesion.

Importante: aunque `grid`, `qualifying_position`, `q1`, `q2` y `q3` aceptan `null`, las claves deben enviarse. El pipeline de inferencia valida que esas columnas existan.

### Response

```json
{
  "race_id": 1180,
  "circuit_id": 39,
  "race_date": "2026-08-23",
  "dashboard_analysis": {
    "top_3": {
      "label": "Predicted podium",
      "requested_size": 3,
      "actual_size": 3,
      "drivers": [
        {
          "predicted_position": 1,
          "driverId": 844,
          "constructorId": 6,
          "score": 2.1352577209
        }
      ],
      "score_summary": {
        "average": 1.9186,
        "min": 1.5957,
        "max": 2.1352,
        "spread": 0.5395
      },
      "shared_top_contributions": [
        {
          "feature": "driver_last_10_top3_rate",
          "mean_contribution": 0.185,
          "total_contribution": 0.555,
          "abs_mean_contribution": 0.185,
          "direction": "up"
        }
      ],
      "feature_group_averages": {
        "starting_position": {
          "grid": 2,
          "qualifying_position": 2,
          "qualifying_gap_to_pole_ms": 180
        },
        "driver_form": {
          "driver_last_10_top3_rate": 0.63,
          "driver_last_10_top10_rate": 0.93,
          "driver_last_10_dnf_rate": 0.03
        }
      }
    },
    "top_5": {
      "label": "Predicted top 5",
      "requested_size": 5,
      "actual_size": 5,
      "drivers": [],
      "score_summary": {},
      "shared_top_contributions": [],
      "feature_group_averages": {}
    },
    "top_10": {
      "label": "Predicted top 10",
      "requested_size": 10,
      "actual_size": 10,
      "drivers": [],
      "score_summary": {},
      "shared_top_contributions": [],
      "feature_group_averages": {}
    }
  },
  "analysis_summary": {
    "participant_count": 2,
    "model_output_note": "score is a relative ranking value; it is not a probability or a causal prediction.",
    "explanation_note": "top_contributions shows how much each feature pushes a driver score inside the XGBoost model for this request.",
    "available_feature_groups": [
      "starting_position",
      "driver_form",
      "driver_circuit_history",
      "constructor_form",
      "constructor_circuit_fit",
      "constructor_pair",
      "circuit_profile"
    ],
    "global_feature_importance": [
      {
        "feature": "driver_last_10_top3_rate",
        "importance": 0.33257326,
        "importance_pct": 0.33257326
      }
    ],
    "model_metrics": {
      "validation": {
        "ndcg": 0.947,
        "spearman": 0.649,
        "mae_position": 3.809,
        "top1_accuracy": 0.506,
        "top3_accuracy": 0.648,
        "top10_accuracy": 0.761
      },
      "test": {
        "ndcg": 0.948,
        "spearman": 0.651,
        "mae_position": 3.394,
        "top1_accuracy": 0.517,
        "top3_accuracy": 0.676,
        "top10_accuracy": 0.775
      }
    }
  },
  "predictions": [
    {
      "predicted_position": 1,
      "driverId": 844,
      "constructorId": 6,
      "score": 1.76,
      "analysis": {
        "feature_groups": {
          "starting_position": {
            "grid": 1,
            "grid_rank_within_race": 1,
            "qualifying_position": 1,
            "qualifying_gap_to_pole_ms": 0,
            "q1_ms": 86572,
            "q2_ms": 85900,
            "q3_ms": 85100
          },
          "driver_form": {
            "driver_points_prev": 250,
            "driver_wins_prev": 5,
            "driver_season_points_before_race": 180,
            "driver_last_3_avg_finish_position": 2.33,
            "driver_last_5_top3_rate": 0.6,
            "driver_last_10_dnf_rate": 0
          },
          "driver_circuit_history": {},
          "constructor_form": {},
          "constructor_circuit_fit": {},
          "constructor_pair": {},
          "circuit_profile": {
            "circuitId": 39,
            "circuit_speed_score": 3,
            "circuit_aero_load_score": 3,
            "circuit_corner_density_score": 3,
            "circuit_is_street": 0
          }
        },
        "top_contributions": [
          {
            "feature": "grid",
            "contribution": 0.4676779807,
            "abs_contribution": 0.4676779807,
            "direction": "up"
          }
        ],
        "bias": 0.1291
      }
    }
  ]
}
```

`predictions` viene ordenado por score descendente. Para la UI, ordenar o mostrar por `predicted_position` ascendente.

`score` es una puntuacion interna del modelo de ranking. Sirve para ordenar pilotos dentro de la misma peticion, no debe mostrarse como probabilidad de victoria.

## Datos de Analisis Devueltos

`dashboard_analysis`:

- `top_3`: analisis agregado del podio predicho.
- `top_5`: analisis agregado de los primeros 5 predichos.
- `top_10`: analisis agregado de los primeros 10 predichos.
- Cada grupo incluye `drivers`, `score_summary`, `shared_top_contributions` y `feature_group_averages`.
- Si la peticion tiene menos pilotos que el tamano solicitado, `actual_size` indica cuantos pilotos reales entraron al grupo.
- Los textos generados dentro de este bloque, como `label`, se devuelven en ingles.

`analysis_summary`:

- `participant_count`: cantidad de pilotos usados en la prediccion.
- `model_output_note`: aviso para UI o tooltips sobre como interpretar `score`.
- `explanation_note`: aviso para UI o tooltips sobre como interpretar contribuciones.
- `available_feature_groups`: grupos de senales que puede renderizar el frontend.
- `global_feature_importance`: top 20 de features mas importantes del entrenamiento completo.
- `model_metrics`: metricas historicas de validacion y test para mostrar calidad del modelo.
- Los textos generados dentro de este bloque se devuelven en ingles.

`predictions[].constructorId`:

- ID del equipo usado por ese piloto en la prediccion.
- Permite unir la respuesta con el catalogo visual del frontend.

`predictions[].analysis.feature_groups`:

- Snapshot de datos usados por el modelo para ese piloto.
- Los valores pueden ser `null` si no hay historico suficiente.
- Estos grupos sirven para tarjetas, tablas comparativas o radar charts.

Grupos disponibles:

- `starting_position`: parrilla, clasificacion, tiempos Q1/Q2/Q3 y gap a la pole.
- `driver_form`: forma reciente del piloto, puntos, victorias, top 3/top 10 y DNF.
- `driver_circuit_history`: historial del piloto en ese circuito o circuitos similares.
- `constructor_form`: forma reciente del constructor/equipo.
- `constructor_circuit_fit`: rendimiento del equipo en ese circuito, circuitos similares y perfiles de velocidad/aero/street.
- `constructor_pair`: rendimiento conjunto reciente del equipo con sus dos autos.
- `circuit_profile`: latitud, longitud, altitud y perfil manual del circuito.

`predictions[].analysis.top_contributions`:

- Top features que mas movieron el `score` de ese piloto.
- `contribution > 0` empuja el score hacia arriba.
- `contribution < 0` empuja el score hacia abajo.
- `direction` es `up` o `down` para pintar badges/colores.
- No son causalidad; son contribuciones internas del modelo para esta fila.

`predictions[].analysis.bias`:

- Valor base del modelo antes de sumar contribuciones.
- Normalmente no hace falta mostrarlo en la UI principal.

## Ideas de Dashboard Profesional

Con la respuesta enriquecida de `/predict`, el frontend puede mostrar:

- ranking predicho con piloto, equipo, posicion y score relativo;
- grafico de barras de `global_feature_importance` para explicar que variables pesan mas en el modelo;
- panel de calidad historica con `model_metrics`;
- top razones por piloto usando `top_contributions`;
- analisis agregado del podio, top 5 y top 10 con `dashboard_analysis`;
- comparativa de forma reciente con `driver_form` y `constructor_form`;
- tarjeta de clasificacion/parrilla con `starting_position`;
- tarjeta de ajuste al circuito con `constructor_circuit_fit` y `circuit_profile`;
- indicadores de confiabilidad con tasas DNF de piloto y constructor;
- comparativa entre companeros de equipo con `constructor_pair`.

## Errores de POST /predict

`422 Unprocessable Entity`:

- JSON invalido o faltan campos requeridos por Pydantic.
- `participants` tiene menos de 2 elementos.
- Tipos incompatibles, por ejemplo `driverId` como texto no convertible a entero.

Ejemplo:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "race_id"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

`400 Bad Request`:

- Faltan archivos del modelo o datos requeridos.
- `race_date` no fue enviado y `race_id` no existe en `races.csv`.
- El pipeline no puede construir features para la carrera.

Ejemplo:

```json
{
  "detail": "race_date is required when raceId does not exist in races.csv."
}
```

## Contrato TypeScript Sugerido

```ts
export type PredictionParticipant = {
  driverId: number;
  constructorId: number;
  grid: number | null;
  qualifying_position: number | null;
  q1: string | number | null;
  q2: string | number | null;
  q3: string | number | null;
};

export type PredictionRequest = {
  race_id: number;
  circuit_id: number;
  race_date?: string | null;
  participants: PredictionParticipant[];
};

export type FeatureValue = string | number | boolean | null;

export type FeatureGroupName =
  | "starting_position"
  | "driver_form"
  | "driver_circuit_history"
  | "constructor_form"
  | "constructor_circuit_fit"
  | "constructor_pair"
  | "circuit_profile";

export type FeatureGroupValues = Record<string, FeatureValue>;

export type GlobalFeatureImportance = {
  feature: string;
  importance: number;
  importance_pct: number;
};

export type FeatureContribution = {
  feature: string;
  contribution: number;
  abs_contribution: number;
  direction: "up" | "down";
};

export type GroupFeatureContribution = {
  feature: string;
  mean_contribution: number;
  total_contribution: number;
  abs_mean_contribution: number;
  direction: "up" | "down";
};

export type DashboardDriver = {
  predicted_position: number;
  driverId: number;
  constructorId: number;
  score: number;
};

export type DashboardScoreSummary = {
  average: number | null;
  min: number | null;
  max: number | null;
  spread: number;
};

export type DashboardGroupAnalysis = {
  label: string;
  requested_size: number;
  actual_size: number;
  drivers: DashboardDriver[];
  score_summary: DashboardScoreSummary;
  shared_top_contributions: GroupFeatureContribution[];
  feature_group_averages: Partial<Record<FeatureGroupName, FeatureGroupValues>>;
};

export type DashboardAnalysis = {
  top_3: DashboardGroupAnalysis;
  top_5: DashboardGroupAnalysis;
  top_10: DashboardGroupAnalysis;
};

export type PredictionAnalysisSummary = {
  participant_count: number;
  model_output_note: string;
  explanation_note: string;
  global_feature_importance: GlobalFeatureImportance[];
  model_metrics: Record<string, Record<string, number>>;
  available_feature_groups: FeatureGroupName[];
};

export type PredictionAnalysis = {
  feature_groups: Partial<Record<FeatureGroupName, FeatureGroupValues>>;
  top_contributions: FeatureContribution[];
  bias: number;
};

export type PredictionItem = {
  predicted_position: number;
  driverId: number;
  constructorId: number;
  score: number;
  analysis: PredictionAnalysis;
};

export type PredictionResponse = {
  race_id: number;
  circuit_id: number;
  race_date: string | null;
  dashboard_analysis: DashboardAnalysis;
  analysis_summary: PredictionAnalysisSummary;
  predictions: PredictionItem[];
};
```

## Validaciones Recomendadas en Frontend

- Exigir al menos 2 participantes.
- Evitar `driverId` duplicados dentro de la misma carrera.
- Enviar todos los participantes con `driverId` y `constructorId`.
- Si la carrera es nueva para el backend, exigir `race_date`.
- Normalizar fechas a `YYYY-MM-DD`.
- Normalizar tiempos de clasificacion a `M:SS.mmm` o enviar `null`.
- No interpretar `score` como porcentaje.
- No presentar `top_contributions` como causalidad garantizada.
- Tratar valores `null` como "sin historico suficiente" o "dato no disponible".
- Mostrar un estado claro cuando `/health` falle.

## Ejemplo Fetch

```ts
const response = await fetch(`${API_BASE_URL}/predict`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    race_id: 1180,
    circuit_id: 39,
    race_date: "2026-08-23",
    participants,
  }),
});

const body = await response.json();

if (!response.ok) {
  throw new Error(
    typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)
  );
}

return body as PredictionResponse;
```

## Datos que el Frontend Debe Tener

Para construir una prediccion usable, el frontend necesita:

- lista de pilotos con su `driverId` y nombre visible;
- lista de equipos con su `constructorId` y nombre visible;
- lista de circuitos con su `circuit_id` y nombre visible;
- calendario o formulario para obtener `race_id`, `circuit_id` y `race_date`;
- resultados de clasificacion/parrilla para llenar `grid`, `qualifying_position`, `q1`, `q2`, `q3`.

Esta API solo predice rankings con IDs y datos numericos/temporales. No traduce nombres a IDs.
