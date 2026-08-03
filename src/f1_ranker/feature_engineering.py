"""Feature engineering historico sin data leakage."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from math import sqrt
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "positionOrder"
GROUP_COLUMN = "raceId"
DATE_COLUMN = "race_date"

RAW_RESULT_COLUMNS = [
    "raceId",
    "driverId",
    "constructorId",
    "grid",
    "rank",
    "points",
    "positionOrder",
]

MODEL_FEATURE_COLUMNS = [
    "driverId",
    "constructorId",
    "grid",
    "qualifying_position",
    "q1_ms",
    "q2_ms",
    "q3_ms",
    "qualifying_gap_to_pole_ms",
    "grid_rank_within_race",
    "circuitId",
    "circuit_lat",
    "circuit_lng",
    "circuit_alt",
    "driver_circuit_pit_time_mean_hist",
    "driver_circuit_pit_time_total_hist",
    "driver_circuit_pit_stop_count_mean_hist",
    "driver_circuit_pit_stop_count_max_hist",
    "driver_circuit_pit_stop_count_min_hist",
    "driver_circuit_lap_time_mean_hist",
    "driver_circuit_race_lap_time_total_mean_hist",
    "driver_circuit_best_lap_hist",
    "driver_circuit_worst_lap_hist",
    "driver_circuit_lap_time_std_hist",
    "driver_circuit_laps_completed_mean_hist",
    "driver_points_prev",
    "driver_wins_prev",
    "driver_championship_position_mean_hist",
    "driver_circuit_wins_hist",
    "driver_circuit_finish_position_mean_hist",
    "driver_last_3_avg_finish_position",
    "driver_last_5_avg_finish_position",
    "driver_last_10_avg_finish_position",
    "driver_last_3_top3_rate",
    "driver_last_5_top3_rate",
    "driver_last_10_top3_rate",
    "driver_last_3_top10_rate",
    "driver_last_5_top10_rate",
    "driver_last_10_top10_rate",
    "driver_season_points_before_race",
    "constructor_points_prev",
    "constructor_wins_prev",
    "constructor_championship_position_mean_hist",
    "constructor_circuit_finish_position_mean_hist",
    "constructor_circuit_wins_hist",
    "constructor_last_3_avg_finish_position",
    "constructor_last_5_avg_finish_position",
    "constructor_last_10_avg_finish_position",
    "constructor_last_3_top10_rate",
    "constructor_last_5_top10_rate",
    "constructor_last_10_top10_rate",
    "constructor_season_points_before_race",
]

CATEGORICAL_FEATURES = ["circuitId"]
NUMERIC_FEATURES = [
    column for column in MODEL_FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES
]


@dataclass
class RunningStats:
    """Estadisticas incrementales para evitar usar informacion futura."""

    count: int = 0
    total: float = 0.0
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        if pd.isna(value):
            return

        value = float(value)
        self.count += 1
        self.total += value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

        delta = value - self.mean_value
        self.mean_value += delta / self.count
        delta_2 = value - self.mean_value
        self.m2 += delta * delta_2

    @property
    def mean(self) -> float:
        return np.nan if self.count == 0 else self.mean_value

    @property
    def std(self) -> float:
        if self.count < 2:
            return np.nan
        return sqrt(self.m2 / (self.count - 1))

    @property
    def minimum(self) -> float:
        return np.nan if self.min_value is None else self.min_value

    @property
    def maximum(self) -> float:
        return np.nan if self.max_value is None else self.max_value


def parse_lap_time_to_ms(value: object) -> float:
    """Convierte tiempos tipo '1:26.572' a milisegundos."""
    if pd.isna(value):
        return np.nan

    text = str(value)
    if ":" not in text:
        return np.nan

    minutes, seconds = text.split(":", maxsplit=1)
    return (int(minutes) * 60 + float(seconds)) * 1000


def _safe_numeric(dataframe: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for column in columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe


def _best_qualifying_time(dataframe: pd.DataFrame) -> pd.Series:
    """Usa Q3 si existe; si no, Q2; si no, Q1."""
    return dataframe["q3_ms"].combine_first(dataframe["q2_ms"]).combine_first(
        dataframe["q1_ms"]
    )


def add_pre_race_relative_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Agrega features disponibles antes de la carrera dentro de cada raceId."""
    dataframe = dataframe.copy()
    dataframe["best_qualifying_ms"] = _best_qualifying_time(dataframe)
    pole_time = dataframe.groupby("raceId")["best_qualifying_ms"].transform("min")
    dataframe["qualifying_gap_to_pole_ms"] = dataframe["best_qualifying_ms"] - pole_time
    dataframe["grid_rank_within_race"] = dataframe.groupby("raceId")["grid"].rank(
        method="min",
        ascending=True,
    )
    return dataframe.drop(columns=["best_qualifying_ms"])


def build_base_dataset(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une tablas disponibles antes de la carrera sin generar duplicados."""
    results = datasets["results"][RAW_RESULT_COLUMNS].copy()
    results = _safe_numeric(
        results,
        [
            "raceId",
            "driverId",
            "constructorId",
            "grid",
            "rank",
            "points",
            "positionOrder",
        ],
    )

    races = datasets["races"][["raceId", "year", "round", "circuitId", "date"]].copy()
    races["race_date"] = pd.to_datetime(races["date"], errors="coerce")
    races = races.drop(columns=["date"])

    circuits = datasets["circuits"][["circuitId", "lat", "lng", "alt"]].rename(
        columns={"lat": "circuit_lat", "lng": "circuit_lng", "alt": "circuit_alt"}
    )
    circuits = _safe_numeric(circuits, ["circuitId", "circuit_lat", "circuit_lng", "circuit_alt"])

    qualifying = datasets["qualifying"][
        ["raceId", "driverId", "constructorId", "position", "q1", "q2", "q3"]
    ].copy()
    qualifying = qualifying.rename(columns={"position": "qualifying_position"})
    qualifying["q1_ms"] = qualifying["q1"].map(parse_lap_time_to_ms)
    qualifying["q2_ms"] = qualifying["q2"].map(parse_lap_time_to_ms)
    qualifying["q3_ms"] = qualifying["q3"].map(parse_lap_time_to_ms)
    qualifying = qualifying.drop(columns=["q1", "q2", "q3"])
    qualifying = _safe_numeric(
        qualifying,
        ["raceId", "driverId", "constructorId", "qualifying_position"],
    )
    qualifying = qualifying.drop_duplicates(
        subset=["raceId", "driverId", "constructorId"], keep="first"
    )

    base = results.merge(races, on="raceId", how="left", validate="many_to_one")
    base = base.merge(circuits, on="circuitId", how="left", validate="many_to_one")
    base = base.merge(
        qualifying,
        on=["raceId", "driverId", "constructorId"],
        how="left",
        validate="one_to_one",
    )
    base = add_pre_race_relative_features(base)

    if base.duplicated(["raceId", "driverId"]).any():
        raise ValueError("El dataset base genero duplicados por raceId + driverId.")

    return base.sort_values([DATE_COLUMN, "raceId", TARGET_COLUMN]).reset_index(drop=True)


def _race_lookup(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    races = datasets["races"][["raceId", "circuitId", "date"]].copy()
    races["race_date"] = pd.to_datetime(races["date"], errors="coerce")
    return races.drop(columns=["date"])


def _pit_summaries(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pit_stops = datasets["pit_stops"][["raceId", "driverId", "milliseconds"]].copy()
    pit_stops = _safe_numeric(pit_stops, ["raceId", "driverId", "milliseconds"])
    race_lookup = _race_lookup(datasets)

    summary = (
        pit_stops.groupby(["raceId", "driverId"], as_index=False)
        .agg(
            pit_time_mean=("milliseconds", "mean"),
            pit_time_total=("milliseconds", "sum"),
            pit_stop_count=("milliseconds", "count"),
        )
        .merge(race_lookup, on="raceId", how="left", validate="many_to_one")
    )
    return summary


def _recent_average(values: deque[float], window: int) -> float:
    recent = list(values)[-window:]
    return np.nan if not recent else float(np.mean(recent))


def _recent_rate(values: deque[float], window: int, threshold: int) -> float:
    recent = list(values)[-window:]
    if not recent:
        return np.nan
    return float(np.mean([value <= threshold for value in recent]))


def _recent_driver_features(history: deque[float], prefix: str) -> dict[str, float]:
    features = {}
    for window in (3, 5, 10):
        features[f"{prefix}_last_{window}_avg_finish_position"] = _recent_average(
            history, window
        )
        features[f"{prefix}_last_{window}_top3_rate"] = _recent_rate(
            history, window, 3
        )
        features[f"{prefix}_last_{window}_top10_rate"] = _recent_rate(
            history, window, 10
        )
    return features


def _recent_constructor_features(history: deque[float]) -> dict[str, float]:
    features = {}
    for window in (3, 5, 10):
        features[f"constructor_last_{window}_avg_finish_position"] = _recent_average(
            history, window
        )
        features[f"constructor_last_{window}_top10_rate"] = _recent_rate(
            history, window, 10
        )
    return features


def _lap_summaries(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    lap_times = datasets["lap_times"][["raceId", "driverId", "milliseconds"]].copy()
    lap_times = _safe_numeric(lap_times, ["raceId", "driverId", "milliseconds"])
    race_lookup = _race_lookup(datasets)

    summary = (
        lap_times.groupby(["raceId", "driverId"], as_index=False)
        .agg(
            lap_time_total=("milliseconds", "sum"),
            laps_completed=("milliseconds", "count"),
            lap_time_mean=("milliseconds", "mean"),
            best_lap=("milliseconds", "min"),
            worst_lap=("milliseconds", "max"),
            lap_time_std=("milliseconds", "std"),
        )
        .merge(race_lookup, on="raceId", how="left", validate="many_to_one")
    )
    return summary


def add_historical_features(
    base: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Agrega historicos calculados solo con carreras anteriores."""
    featured = base.copy()

    pit_by_race = {race_id: rows for race_id, rows in _pit_summaries(datasets).groupby("raceId")}
    lap_by_race = {race_id: rows for race_id, rows in _lap_summaries(datasets).groupby("raceId")}

    driver_standings = _safe_numeric(
        datasets["driver_standings"][["raceId", "driverId", "points", "position", "wins"]].copy(),
        ["raceId", "driverId", "points", "position", "wins"],
    )
    driver_standings_by_race = {
        race_id: rows for race_id, rows in driver_standings.groupby("raceId")
    }

    constructor_standings = _safe_numeric(
        datasets["constructor_standings"][
            ["raceId", "constructorId", "points", "position", "wins"]
        ].copy(),
        ["raceId", "constructorId", "points", "position", "wins"],
    )
    constructor_standings_by_race = {
        race_id: rows for race_id, rows in constructor_standings.groupby("raceId")
    }

    results_by_race = {
        race_id: rows for race_id, rows in featured.groupby("raceId", sort=False)
    }

    pit_duration_stats: dict[tuple[int, int], RunningStats] = {}
    pit_total_stats: dict[tuple[int, int], RunningStats] = {}
    pit_count_stats: dict[tuple[int, int], RunningStats] = {}
    lap_mean_stats: dict[tuple[int, int], RunningStats] = {}
    lap_best_stats: dict[tuple[int, int], RunningStats] = {}
    lap_worst_stats: dict[tuple[int, int], RunningStats] = {}
    lap_std_stats: dict[tuple[int, int], RunningStats] = {}
    lap_race_total_stats: dict[tuple[int, int], RunningStats] = {}
    lap_count_stats: dict[tuple[int, int], RunningStats] = {}
    driver_position_stats: dict[int, RunningStats] = {}
    driver_circuit_finish_stats: dict[tuple[int, int], RunningStats] = {}
    constructor_position_stats: dict[int, RunningStats] = {}
    constructor_circuit_finish_stats: dict[tuple[int, int], RunningStats] = {}

    driver_latest_points: dict[int, float] = {}
    driver_latest_wins: dict[int, float] = {}
    constructor_latest_points: dict[int, float] = {}
    constructor_latest_wins: dict[int, float] = {}
    driver_circuit_wins: dict[tuple[int, int], int] = {}
    constructor_circuit_wins: dict[tuple[int, int], int] = {}
    driver_recent_finishes: dict[int, deque[float]] = {}
    constructor_recent_finishes: dict[int, deque[float]] = {}
    driver_season_points: dict[tuple[int, int], float] = {}
    constructor_season_points: dict[tuple[int, int], float] = {}

    feature_rows = []
    race_order = (
        featured[["raceId", DATE_COLUMN]]
        .drop_duplicates()
        .sort_values([DATE_COLUMN, "raceId"])["raceId"]
        .tolist()
    )

    for race_id in race_order:
        race_rows = results_by_race[race_id]
        for row in race_rows.itertuples():
            driver_id = int(row.driverId)
            constructor_id = int(row.constructorId)
            circuit_id = int(row.circuitId)
            year = int(row.year)
            driver_circuit_key = (driver_id, circuit_id)
            constructor_circuit_key = (constructor_id, circuit_id)
            driver_season_key = (driver_id, year)
            constructor_season_key = (constructor_id, year)

            pit_duration = pit_duration_stats.get(driver_circuit_key, RunningStats())
            pit_total = pit_total_stats.get(driver_circuit_key, RunningStats())
            pit_count = pit_count_stats.get(driver_circuit_key, RunningStats())
            lap_mean = lap_mean_stats.get(driver_circuit_key, RunningStats())
            lap_best = lap_best_stats.get(driver_circuit_key, RunningStats())
            lap_worst = lap_worst_stats.get(driver_circuit_key, RunningStats())
            lap_std = lap_std_stats.get(driver_circuit_key, RunningStats())
            lap_total = lap_race_total_stats.get(driver_circuit_key, RunningStats())
            lap_count = lap_count_stats.get(driver_circuit_key, RunningStats())
            driver_pos = driver_position_stats.get(driver_id, RunningStats())
            driver_circuit_finish = driver_circuit_finish_stats.get(
                driver_circuit_key, RunningStats()
            )
            constructor_pos = constructor_position_stats.get(
                constructor_id, RunningStats()
            )
            constructor_circuit_finish = constructor_circuit_finish_stats.get(
                constructor_circuit_key, RunningStats()
            )

            row_features = {
                    "raceId": race_id,
                    "driverId": driver_id,
                    "driver_circuit_pit_time_mean_hist": pit_duration.mean,
                    "driver_circuit_pit_time_total_hist": pit_total.total
                    if pit_total.count
                    else np.nan,
                    "driver_circuit_pit_stop_count_mean_hist": pit_count.mean,
                    "driver_circuit_pit_stop_count_max_hist": pit_count.maximum,
                    "driver_circuit_pit_stop_count_min_hist": pit_count.minimum,
                    "driver_circuit_lap_time_mean_hist": lap_mean.mean,
                    "driver_circuit_race_lap_time_total_mean_hist": lap_total.mean,
                    "driver_circuit_best_lap_hist": lap_best.minimum,
                    "driver_circuit_worst_lap_hist": lap_worst.maximum,
                    "driver_circuit_lap_time_std_hist": lap_std.mean,
                    "driver_circuit_laps_completed_mean_hist": lap_count.mean,
                    "driver_points_prev": driver_latest_points.get(driver_id, np.nan),
                    "driver_wins_prev": driver_latest_wins.get(driver_id, np.nan),
                    "driver_championship_position_mean_hist": driver_pos.mean,
                    "driver_circuit_wins_hist": driver_circuit_wins.get(
                        driver_circuit_key, 0
                    ),
                    "driver_circuit_finish_position_mean_hist": driver_circuit_finish.mean,
                    "driver_season_points_before_race": driver_season_points.get(
                        driver_season_key, 0.0
                    ),
                    "constructor_points_prev": constructor_latest_points.get(
                        constructor_id, np.nan
                    ),
                    "constructor_wins_prev": constructor_latest_wins.get(
                        constructor_id, np.nan
                    ),
                    "constructor_championship_position_mean_hist": constructor_pos.mean,
                    "constructor_circuit_finish_position_mean_hist": constructor_circuit_finish.mean,
                    "constructor_circuit_wins_hist": constructor_circuit_wins.get(
                        constructor_circuit_key, 0
                    ),
                    "constructor_season_points_before_race": constructor_season_points.get(
                        constructor_season_key, 0.0
                    ),
                }
            row_features.update(
                _recent_driver_features(
                    driver_recent_finishes.get(driver_id, deque(maxlen=10)),
                    "driver",
                )
            )
            row_features.update(
                _recent_constructor_features(
                    constructor_recent_finishes.get(constructor_id, deque(maxlen=20))
                )
            )
            feature_rows.append(row_features)

        for row in pit_by_race.get(race_id, pd.DataFrame()).itertuples():
            key = (int(row.driverId), int(row.circuitId))
            pit_duration_stats.setdefault(key, RunningStats()).add(row.pit_time_mean)
            pit_total_stats.setdefault(key, RunningStats()).add(row.pit_time_total)
            pit_count_stats.setdefault(key, RunningStats()).add(row.pit_stop_count)

        for row in lap_by_race.get(race_id, pd.DataFrame()).itertuples():
            key = (int(row.driverId), int(row.circuitId))
            lap_mean_stats.setdefault(key, RunningStats()).add(row.lap_time_mean)
            lap_best_stats.setdefault(key, RunningStats()).add(row.best_lap)
            lap_worst_stats.setdefault(key, RunningStats()).add(row.worst_lap)
            lap_std_stats.setdefault(key, RunningStats()).add(row.lap_time_std)
            lap_race_total_stats.setdefault(key, RunningStats()).add(row.lap_time_total)
            lap_count_stats.setdefault(key, RunningStats()).add(row.laps_completed)

        for row in driver_standings_by_race.get(race_id, pd.DataFrame()).itertuples():
            driver_id = int(row.driverId)
            driver_latest_points[driver_id] = row.points
            driver_latest_wins[driver_id] = row.wins
            driver_position_stats.setdefault(driver_id, RunningStats()).add(row.position)

        for row in constructor_standings_by_race.get(race_id, pd.DataFrame()).itertuples():
            constructor_id = int(row.constructorId)
            constructor_latest_points[constructor_id] = row.points
            constructor_latest_wins[constructor_id] = row.wins
            constructor_position_stats.setdefault(constructor_id, RunningStats()).add(
                row.position
            )

        for row in race_rows.itertuples():
            driver_key = (int(row.driverId), int(row.circuitId))
            constructor_key = (int(row.constructorId), int(row.circuitId))
            driver_id = int(row.driverId)
            constructor_id = int(row.constructorId)
            year = int(row.year)
            driver_circuit_finish_stats.setdefault(driver_key, RunningStats()).add(
                row.positionOrder
            )
            constructor_circuit_finish_stats.setdefault(constructor_key, RunningStats()).add(
                row.positionOrder
            )
            if row.positionOrder == 1:
                driver_circuit_wins[driver_key] = driver_circuit_wins.get(driver_key, 0) + 1
                constructor_circuit_wins[constructor_key] = (
                    constructor_circuit_wins.get(constructor_key, 0) + 1
                )
            driver_recent_finishes.setdefault(driver_id, deque(maxlen=10)).append(
                row.positionOrder
            )
            constructor_recent_finishes.setdefault(
                constructor_id, deque(maxlen=20)
            ).append(row.positionOrder)
            driver_season_key = (driver_id, year)
            constructor_season_key = (constructor_id, year)
            points = 0.0 if pd.isna(row.points) else float(row.points)
            driver_season_points[driver_season_key] = (
                driver_season_points.get(driver_season_key, 0.0) + points
            )
            constructor_season_points[constructor_season_key] = (
                constructor_season_points.get(constructor_season_key, 0.0) + points
            )

    history = pd.DataFrame(feature_rows)
    featured = featured.merge(history, on=["raceId", "driverId"], how="left", validate="one_to_one")
    return featured


def build_model_dataset(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Construye el dataset final con una fila por piloto y carrera."""
    base = build_base_dataset(datasets)
    dataset = add_historical_features(base, datasets)
    dataset["circuitId"] = dataset["circuitId"].astype("Int64").astype("category")

    if dataset.duplicated(["raceId", "driverId"]).any():
        raise ValueError("El dataset final contiene duplicados por raceId + driverId.")

    return dataset


def build_preprocessor() -> ColumnTransformer:
    """Crea transformaciones reutilizables para entrenamiento e inferencia."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def save_feature_artifacts(preprocessor: ColumnTransformer, path: Path) -> None:
    """Guarda las transformaciones para reutilizarlas en inferencia."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_feature_artifacts(path: Path) -> ColumnTransformer:
    """Carga las transformaciones usadas durante entrenamiento."""
    return joblib.load(path)
