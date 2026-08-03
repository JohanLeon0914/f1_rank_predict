"""Inferencia para nuevas carreras."""

from pathlib import Path

import numpy as np
import pandas as pd

from f1_ranker.config import MODEL_DIR
from f1_ranker.data_loading import load_training_data
from f1_ranker.feature_engineering import (
    MODEL_FEATURE_COLUMNS,
    add_historical_features,
    build_base_dataset,
)
from f1_ranker.training import load_trained_ranker


def _resolve_race_date(
    race_id: int,
    race_date: str | None,
    races: pd.DataFrame,
) -> pd.Timestamp:
    if race_date is not None:
        return pd.to_datetime(race_date)

    match = races.loc[races["raceId"] == race_id, "date"]
    if match.empty:
        raise ValueError(
            "race_date es obligatorio cuando raceId no existe en races.csv."
        )
    return pd.to_datetime(match.iloc[0])


def build_inference_dataset(
    race_id: int,
    circuit_id: int,
    participants: pd.DataFrame,
    datasets: dict[str, pd.DataFrame] | None = None,
    race_date: str | None = None,
) -> pd.DataFrame:
    """Construye features para una carrera nueva usando solo datos anteriores.

    `participants` debe contener driverId, constructorId, grid,
    qualifying_position, q1, q2 y q3. Los tiempos q1/q2/q3 aceptan formato
    '1:26.572' o valores faltantes.
    """
    datasets = load_training_data() if datasets is None else datasets
    target_date = _resolve_race_date(race_id, race_date, datasets["races"])

    required = {"driverId", "constructorId", "grid", "qualifying_position", "q1", "q2", "q3"}
    missing = required - set(participants.columns)
    if missing:
        raise ValueError(f"Faltan columnas en participants: {sorted(missing)}")

    races = datasets["races"].copy()
    races["race_date"] = pd.to_datetime(races["date"], errors="coerce")
    previous_race_ids = set(races.loc[races["race_date"] < target_date, "raceId"])

    historical_datasets = datasets.copy()
    historical_datasets["results"] = datasets["results"][
        datasets["results"]["raceId"].isin(previous_race_ids)
    ].copy()

    synthetic_results = participants[["driverId", "constructorId", "grid"]].copy()
    synthetic_results["raceId"] = race_id
    synthetic_results["rank"] = np.nan
    synthetic_results["points"] = np.nan
    synthetic_results["positionOrder"] = np.nan
    synthetic_results = synthetic_results[
        [
            "raceId",
            "driverId",
            "constructorId",
            "grid",
            "rank",
            "points",
            "positionOrder",
        ]
    ]

    synthetic_race = pd.DataFrame(
        [
            {
                "raceId": race_id,
                "year": target_date.year,
                "round": np.nan,
                "circuitId": circuit_id,
                "date": target_date.strftime("%Y-%m-%d"),
            }
        ]
    )

    synthetic_qualifying = participants[
        ["driverId", "constructorId", "qualifying_position", "q1", "q2", "q3"]
    ].copy()
    synthetic_qualifying["raceId"] = race_id
    synthetic_qualifying = synthetic_qualifying.rename(
        columns={"qualifying_position": "position"}
    )
    synthetic_qualifying = synthetic_qualifying[
        ["raceId", "driverId", "constructorId", "position", "q1", "q2", "q3"]
    ]

    historical_datasets["results"] = pd.concat(
        [historical_datasets["results"], synthetic_results], ignore_index=True
    )
    historical_datasets["races"] = pd.concat(
        [
            datasets["races"][datasets["races"]["raceId"].isin(previous_race_ids)],
            synthetic_race,
        ],
        ignore_index=True,
    )
    historical_datasets["qualifying"] = pd.concat(
        [datasets["qualifying"], synthetic_qualifying], ignore_index=True
    )

    base = build_base_dataset(historical_datasets)
    featured = add_historical_features(base, historical_datasets)
    inference_rows = featured[featured["raceId"] == race_id].copy()
    inference_rows["circuitId"] = inference_rows["circuitId"].astype("Int64").astype("category")

    return inference_rows


def predict_race_ranking(
    race_id: int,
    circuit_id: int,
    participants: pd.DataFrame,
    model_dir: Path = MODEL_DIR,
    race_date: str | None = None,
) -> pd.DataFrame:
    """Devuelve ranking predicho ordenado por score descendente."""
    model, artifacts = load_trained_ranker(model_dir)
    dataset = build_inference_dataset(
        race_id=race_id,
        circuit_id=circuit_id,
        participants=participants,
        race_date=race_date,
    )

    preprocessor = artifacts["preprocessor"]
    x_matrix = preprocessor.transform(dataset[MODEL_FEATURE_COLUMNS])
    scores = model.predict(x_matrix)

    prediction = pd.DataFrame(
        {
            "DriverId": dataset["driverId"].to_numpy(),
            "Score": scores,
        }
    ).sort_values("Score", ascending=False)
    prediction.insert(0, "Posicion Predicha", range(1, len(prediction) + 1))

    return prediction.reset_index(drop=True)
