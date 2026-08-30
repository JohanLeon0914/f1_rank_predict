"""API HTTP para servir predicciones del modelo F1 Ranker."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from f1_ranker.config import MODEL_DIR
from f1_ranker.inference import predict_race_ranking_with_analysis


class ParticipantRequest(BaseModel):
    """Datos disponibles antes de una carrera para un piloto."""

    driverId: int = Field(..., description="Driver identifier.")
    constructorId: int = Field(..., description="Constructor/team identifier.")
    grid: int | float | None = Field(None, description="Starting grid position.")
    qualifying_position: int | float | None = Field(None, description="Qualifying position.")
    q1: str | float | None = Field(None, description="Q1 lap time in M:SS.mmm format.")
    q2: str | float | None = Field(None, description="Q2 lap time in M:SS.mmm format.")
    q3: str | float | None = Field(None, description="Q3 lap time in M:SS.mmm format.")


class PredictionRequest(BaseModel):
    """Contrato de entrada para predecir el ranking de una carrera."""

    race_id: int = Field(..., description="Race identifier to predict.")
    circuit_id: int = Field(..., description="Circuit identifier.")
    race_date: str | None = Field(
        None,
        description="YYYY-MM-DD date. Required when race_id does not exist in races.csv.",
    )
    participants: list[ParticipantRequest] = Field(
        ...,
        min_length=2,
        description="Drivers participating in the race.",
    )


class PredictionItem(BaseModel):
    predicted_position: int
    driverId: int
    constructorId: int
    score: float
    analysis: dict[str, Any]


class PredictionResponse(BaseModel):
    race_id: int
    circuit_id: int
    race_date: str | None
    dashboard_analysis: dict[str, Any]
    analysis_summary: dict[str, Any]
    predictions: list[PredictionItem]


app = FastAPI(
    title="F1 Ranker API",
    version="1.0.0",
    description="API for predicting a Formula 1 race ranking.",
)


def _json_safe(value: Any) -> Any:
    """Convierte NaN a None para respuestas JSON validas."""
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _json_safe(exc.errors())})


@lru_cache(maxsize=1)
def _load_metrics() -> dict[str, Any]:
    metrics_path = MODEL_DIR / "metrics.joblib"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No existen metricas entrenadas en {metrics_path}")
    return joblib.load(metrics_path)


@app.get("/health")
def health() -> dict[str, str]:
    """Verifica que los artefactos minimos del modelo existan."""
    required_files = [
        MODEL_DIR / "xgb_ranker.json",
        MODEL_DIR / "feature_artifacts.joblib",
        MODEL_DIR / "metrics.joblib",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise HTTPException(status_code=503, detail={"missing_files": missing})
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Devuelve las metricas guardadas del ultimo entrenamiento."""
    try:
        return _load_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predice el orden final esperado para los pilotos enviados."""
    participants = pd.DataFrame([participant.model_dump() for participant in request.participants])

    try:
        prediction = predict_race_ranking_with_analysis(
            race_id=request.race_id,
            circuit_id=request.circuit_id,
            participants=participants,
            race_date=request.race_date,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(**_json_safe(prediction))
