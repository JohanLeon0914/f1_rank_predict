"""Inferencia para nuevas carreras."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from f1_ranker.config import MODEL_DIR
from f1_ranker.data_loading import load_training_data
from f1_ranker.feature_engineering import (
    MODEL_FEATURE_COLUMNS,
    add_historical_features,
    build_base_dataset,
)
from f1_ranker.training import load_trained_ranker


EXPLANATION_FEATURES = {
    "starting_position": [
        "grid",
        "grid_rank_within_race",
        "qualifying_position",
        "qualifying_gap_to_pole_ms",
        "q1_ms",
        "q2_ms",
        "q3_ms",
    ],
    "driver_form": [
        "driver_points_prev",
        "driver_wins_prev",
        "driver_season_points_before_race",
        "driver_last_3_avg_finish_position",
        "driver_last_5_avg_finish_position",
        "driver_last_10_avg_finish_position",
        "driver_last_3_top3_rate",
        "driver_last_5_top3_rate",
        "driver_last_10_top3_rate",
        "driver_last_3_top10_rate",
        "driver_last_5_top10_rate",
        "driver_last_10_top10_rate",
        "driver_last_3_dnf_rate",
        "driver_last_5_dnf_rate",
        "driver_last_10_dnf_rate",
    ],
    "driver_circuit_history": [
        "driver_circuit_wins_hist",
        "driver_circuit_finish_position_mean_hist",
        "driver_circuit_dnf_rate_hist",
        "driver_similar_circuit_finish_position_mean_hist",
        "driver_similar_circuit_top10_rate_hist",
        "driver_similar_circuit_dnf_rate_hist",
    ],
    "constructor_form": [
        "constructor_points_prev",
        "constructor_wins_prev",
        "constructor_season_points_before_race",
        "constructor_last_3_avg_finish_position",
        "constructor_last_5_avg_finish_position",
        "constructor_last_10_avg_finish_position",
        "constructor_last_3_top10_rate",
        "constructor_last_5_top10_rate",
        "constructor_last_10_top10_rate",
        "constructor_last_3_dnf_rate",
        "constructor_last_5_dnf_rate",
        "constructor_last_10_dnf_rate",
    ],
    "constructor_circuit_fit": [
        "constructor_circuit_finish_position_mean_hist",
        "constructor_circuit_wins_hist",
        "constructor_circuit_top10_rate_hist",
        "constructor_circuit_dnf_rate_hist",
        "constructor_similar_circuit_finish_position_mean_hist",
        "constructor_similar_circuit_top10_rate_hist",
        "constructor_similar_circuit_dnf_rate_hist",
        "constructor_speed_profile_finish_position_mean_hist",
        "constructor_speed_profile_top10_rate_hist",
        "constructor_speed_profile_dnf_rate_hist",
        "constructor_aero_profile_finish_position_mean_hist",
        "constructor_aero_profile_top10_rate_hist",
        "constructor_aero_profile_dnf_rate_hist",
        "constructor_street_finish_position_mean_hist",
        "constructor_street_top10_rate_hist",
        "constructor_street_dnf_rate_hist",
    ],
    "constructor_pair": [
        "constructor_pair_last_3_avg_finish_position",
        "constructor_pair_last_5_avg_finish_position",
        "constructor_pair_last_3_best_finish_position",
        "constructor_pair_last_5_best_finish_position",
        "constructor_pair_last_3_points",
        "constructor_pair_last_5_points",
        "constructor_pair_last_3_both_top10_rate",
        "constructor_pair_last_5_both_top10_rate",
        "constructor_pair_last_3_double_dnf_rate",
        "constructor_pair_last_5_double_dnf_rate",
    ],
    "circuit_profile": [
        "circuitId",
        "circuit_lat",
        "circuit_lng",
        "circuit_alt",
        "circuit_speed_score",
        "circuit_aero_load_score",
        "circuit_corner_density_score",
        "circuit_is_street",
    ],
}

TOP_GROUPS = {
    "top_3": "Predicted podium",
    "top_5": "Predicted top 5",
    "top_10": "Predicted top 10",
}


def _clean_transformed_feature_name(name: str) -> str:
    if name.startswith("numeric__"):
        return name.replace("numeric__", "", 1)
    if name.startswith("categorical__"):
        raw_name = name.replace("categorical__", "", 1)
        if raw_name.startswith("circuitId_"):
            return "circuitId"
        return raw_name
    return name


def _safe_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _row_values(row: pd.Series, columns: list[str]) -> dict[str, object]:
    return {
        column: _safe_value(row[column])
        for column in columns
        if column in row.index
    }


def _aggregate_contributions(
    transformed_feature_names: np.ndarray,
    contribution_values: np.ndarray,
) -> list[dict[str, object]]:
    rows = pd.DataFrame(
        {
            "feature": [
                _clean_transformed_feature_name(name) for name in transformed_feature_names
            ],
            "contribution": contribution_values,
        }
    )
    grouped = (
        rows.groupby("feature", as_index=False)["contribution"]
        .sum()
        .assign(abs_contribution=lambda data: data["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .reset_index(drop=True)
    )
    return [
        {
            "feature": str(row.feature),
            "contribution": float(row.contribution),
            "abs_contribution": float(row.abs_contribution),
            "direction": "up" if row.contribution >= 0 else "down",
        }
        for row in grouped.head(12).itertuples()
    ]


def _load_feature_importance(model_dir: Path) -> list[dict[str, object]]:
    path = model_dir / "feature_importance.csv"
    if not path.exists():
        return []
    report = pd.read_csv(path)
    rows = []
    for row in report.head(20).itertuples():
        rows.append(
            {
                "feature": str(row.original_feature),
                "importance": float(row.importance),
                "importance_pct": float(row.importance_pct),
            }
        )
    return rows


def _load_model_metrics(model_dir: Path) -> dict[str, object]:
    path = model_dir / "metrics.joblib"
    if not path.exists():
        return {}
    return joblib.load(path)


def _mean_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _feature_group_averages(rows: pd.DataFrame) -> dict[str, dict[str, object]]:
    averages: dict[str, dict[str, object]] = {}
    for group_name, columns in EXPLANATION_FEATURES.items():
        group_values = {}
        for column in columns:
            if column in rows.columns:
                group_values[column] = _mean_or_none(rows[column])
        averages[group_name] = group_values
    return averages


def _aggregate_group_contributions(
    transformed_feature_names: np.ndarray,
    contribution_rows: np.ndarray,
) -> list[dict[str, object]]:
    raw = pd.DataFrame(
        contribution_rows,
        columns=[
            _clean_transformed_feature_name(name)
            for name in transformed_feature_names
        ],
    )
    grouped = raw.T.groupby(level=0).sum().T

    summary = pd.DataFrame(
        {
            "feature": grouped.columns,
            "mean_contribution": grouped.mean(axis=0).to_numpy(),
            "total_contribution": grouped.sum(axis=0).to_numpy(),
        }
    )
    summary["abs_mean_contribution"] = summary["mean_contribution"].abs()
    summary = summary.sort_values("abs_mean_contribution", ascending=False).reset_index(drop=True)

    return [
        {
            "feature": str(row.feature),
            "mean_contribution": float(row.mean_contribution),
            "total_contribution": float(row.total_contribution),
            "abs_mean_contribution": float(row.abs_mean_contribution),
            "direction": "up" if row.mean_contribution >= 0 else "down",
        }
        for row in summary.head(12).itertuples()
    ]


def _build_dashboard_group_analysis(
    output: pd.DataFrame,
    transformed_feature_names: np.ndarray,
    contributions: np.ndarray,
) -> dict[str, object]:
    dashboard: dict[str, object] = {}

    for group_key, label in TOP_GROUPS.items():
        requested_size = int(group_key.replace("top_", ""))
        group_rows = output.head(requested_size).copy()
        source_indices = group_rows["_source_index"].astype(int).tolist()
        scores = group_rows["score"].astype(float)
        dashboard[group_key] = {
            "label": label,
            "requested_size": requested_size,
            "actual_size": int(len(group_rows)),
            "drivers": [
                {
                    "predicted_position": int(row.predicted_position),
                    "driverId": int(row.driverId),
                    "constructorId": int(row.constructorId),
                    "score": float(row.score),
                }
                for row in group_rows.itertuples()
            ],
            "score_summary": {
                "average": float(scores.mean()) if not scores.empty else None,
                "min": float(scores.min()) if not scores.empty else None,
                "max": float(scores.max()) if not scores.empty else None,
                "spread": float(scores.max() - scores.min()) if len(scores) > 1 else 0.0,
            },
            "shared_top_contributions": _aggregate_group_contributions(
                transformed_feature_names,
                contributions[source_indices, :-1],
            )
            if source_indices
            else [],
            "feature_group_averages": _feature_group_averages(group_rows),
        }

    return dashboard


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
            "race_date is required when raceId does not exist in races.csv."
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
        raise ValueError(f"Missing participant columns: {sorted(missing)}")

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
    synthetic_results["statusId"] = np.nan
    synthetic_results = synthetic_results[
        [
            "raceId",
            "driverId",
            "constructorId",
            "grid",
            "rank",
            "points",
            "positionOrder",
            "statusId",
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


def predict_race_ranking_with_analysis(
    race_id: int,
    circuit_id: int,
    participants: pd.DataFrame,
    model_dir: Path = MODEL_DIR,
    race_date: str | None = None,
) -> dict[str, object]:
    """Devuelve ranking y datos explicativos para dashboards.

    Las contribuciones son valores tipo SHAP emitidos por XGBoost sobre las
    features transformadas y luego agregadas a las features originales.
    """
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

    transformed_feature_names = preprocessor.get_feature_names_out()
    booster = model.get_booster()
    contributions = booster.predict(
        xgb.DMatrix(x_matrix, feature_names=list(transformed_feature_names)),
        pred_contribs=True,
    )

    identity_columns = ["raceId", "driverId", "constructorId", "race_date"]
    output_columns = identity_columns + [
        column for column in MODEL_FEATURE_COLUMNS if column not in identity_columns
    ]
    output = dataset[output_columns].copy()
    output["score"] = scores
    output["_source_index"] = range(len(output))
    output = output.sort_values("score", ascending=False).reset_index(drop=True)
    output["predicted_position"] = range(1, len(output) + 1)

    prediction_rows = []
    for _, row in output.iterrows():
        source_index = int(row["_source_index"])
        row_series = dataset.iloc[source_index]
        feature_groups = {
            group_name: _row_values(row_series, columns)
            for group_name, columns in EXPLANATION_FEATURES.items()
        }
        prediction_rows.append(
            {
                "predicted_position": int(row["predicted_position"]),
                "driverId": int(row["driverId"]),
                "constructorId": int(row["constructorId"]),
                "score": float(row["score"]),
                "analysis": {
                    "feature_groups": feature_groups,
                    "top_contributions": _aggregate_contributions(
                        transformed_feature_names,
                        contributions[source_index][:-1],
                    ),
                    "bias": float(contributions[source_index][-1]),
                },
            }
        )

    first_row = dataset.iloc[0]
    return {
        "race_id": race_id,
        "circuit_id": circuit_id,
        "race_date": _safe_value(first_row["race_date"]),
        "dashboard_analysis": _build_dashboard_group_analysis(
            output,
            transformed_feature_names,
            contributions,
        ),
        "analysis_summary": {
            "participant_count": int(len(dataset)),
            "model_output_note": (
                "score is a relative ranking value; it is not a probability "
                "or a causal prediction."
            ),
            "explanation_note": (
                "top_contributions shows how much each feature pushes a "
                "driver score inside the XGBoost model for this request."
            ),
            "global_feature_importance": _load_feature_importance(model_dir),
            "model_metrics": _load_model_metrics(model_dir),
            "available_feature_groups": list(EXPLANATION_FEATURES.keys()),
        },
        "predictions": prediction_rows,
    }
