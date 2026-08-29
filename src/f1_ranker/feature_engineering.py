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
    "statusId",
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
    "circuit_speed_score",
    "circuit_aero_load_score",
    "circuit_corner_density_score",
    "circuit_is_street",
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
    "driver_last_3_dnf_rate",
    "driver_last_5_dnf_rate",
    "driver_last_10_dnf_rate",
    "driver_last_3_qualy_finish_delta",
    "driver_last_5_qualy_finish_delta",
    "driver_last_10_qualy_finish_delta",
    "driver_season_points_before_race",
    "driver_circuit_dnf_rate_hist",
    "driver_similar_circuit_finish_position_mean_hist",
    "driver_similar_circuit_top10_rate_hist",
    "driver_similar_circuit_dnf_rate_hist",
    "constructor_points_prev",
    "constructor_wins_prev",
    "constructor_championship_position_mean_hist",
    "constructor_circuit_finish_position_mean_hist",
    "constructor_circuit_wins_hist",
    "constructor_circuit_top10_rate_hist",
    "constructor_circuit_dnf_rate_hist",
    "constructor_last_3_avg_finish_position",
    "constructor_last_5_avg_finish_position",
    "constructor_last_10_avg_finish_position",
    "constructor_last_3_top10_rate",
    "constructor_last_5_top10_rate",
    "constructor_last_10_top10_rate",
    "constructor_last_3_dnf_rate",
    "constructor_last_5_dnf_rate",
    "constructor_last_10_dnf_rate",
    "constructor_last_3_qualy_finish_delta",
    "constructor_last_5_qualy_finish_delta",
    "constructor_last_10_qualy_finish_delta",
    "constructor_season_points_before_race",
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
]

CATEGORICAL_FEATURES = ["circuitId"]
NUMERIC_FEATURES = [
    column for column in MODEL_FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES
]

CLASSIFIED_STATUS_IDS = set(range(1, 20))

CIRCUIT_PROFILE_OVERRIDES = {
    "monza": (5, 1, 1, 0),
    "spa": (5, 2, 2, 0),
    "silverstone": (5, 3, 3, 0),
    "hockenheimring": (4, 2, 2, 0),
    "baku": (5, 2, 2, 1),
    "jeddah": (5, 2, 3, 1),
    "vegas": (5, 1, 1, 1),
    "avus": (5, 1, 1, 0),
    "reims": (5, 1, 1, 0),
    "indianapolis": (4, 1, 1, 0),
    "red_bull_ring": (4, 2, 2, 0),
    "fuji": (4, 2, 2, 0),
    "shanghai": (4, 3, 3, 0),
    "bahrain": (4, 3, 3, 0),
    "sepang": (4, 3, 3, 0),
    "catalunya": (3, 4, 4, 0),
    "suzuka": (4, 4, 5, 0),
    "hungaroring": (2, 5, 5, 0),
    "monaco": (1, 5, 5, 1),
    "marina_bay": (2, 5, 5, 1),
    "valencia": (3, 3, 3, 1),
    "miami": (4, 2, 2, 1),
    "las_vegas": (2, 2, 2, 1),
    "detroit": (2, 4, 4, 1),
    "phoenix": (2, 4, 4, 1),
    "long_beach": (2, 4, 4, 1),
    "dallas": (2, 4, 4, 1),
    "adelaide": (2, 4, 4, 1),
    "madring": (3, 3, 3, 1),
    "zandvoort": (3, 5, 5, 0),
    "imola": (3, 4, 4, 0),
    "magny_cours": (3, 4, 4, 0),
    "nurburgring": (3, 4, 4, 0),
    "interlagos": (3, 4, 4, 0),
    "yas_marina": (3, 3, 3, 0),
    "rodriguez": (4, 3, 3, 0),
    "americas": (3, 4, 4, 0),
    "portimao": (3, 4, 4, 0),
    "mugello": (4, 4, 4, 0),
    "losail": (4, 4, 4, 0),
}


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


def _profile_bucket(value: object) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def _circuit_profile_key(row: object) -> tuple[int, int, int]:
    return (
        _profile_bucket(getattr(row, "circuit_speed_score")),
        _profile_bucket(getattr(row, "circuit_aero_load_score")),
        _profile_bucket(getattr(row, "circuit_is_street")),
    )


def _add_circuit_profile_features(circuits: pd.DataFrame) -> pd.DataFrame:
    circuits = circuits.copy()

    profile_rows = []
    for row in circuits.itertuples():
        circuit_ref = str(getattr(row, "circuitRef", "")).strip('"')
        speed, aero, corners, is_street = CIRCUIT_PROFILE_OVERRIDES.get(
            circuit_ref,
            (3, 3, 3, int("street" in circuit_ref.lower())),
        )
        profile_rows.append(
            {
                "circuitId": row.circuitId,
                "circuit_speed_score": speed,
                "circuit_aero_load_score": aero,
                "circuit_corner_density_score": corners,
                "circuit_is_street": is_street,
            }
        )

    return circuits.merge(pd.DataFrame(profile_rows), on="circuitId", how="left")


def _is_classified_finish(status_id: object) -> bool:
    if pd.isna(status_id):
        return False
    return int(status_id) in CLASSIFIED_STATUS_IDS


def _qualy_finish_delta(position_order: object, qualifying_position: object) -> float:
    if pd.isna(position_order) or pd.isna(qualifying_position):
        return np.nan
    return float(position_order) - float(qualifying_position)


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
            "statusId",
        ],
    )

    races = datasets["races"][["raceId", "year", "round", "circuitId", "date"]].copy()
    races["race_date"] = pd.to_datetime(races["date"], errors="coerce")
    races = races.drop(columns=["date"])

    circuits = datasets["circuits"][
        ["circuitId", "circuitRef", "lat", "lng", "alt"]
    ].rename(
        columns={"lat": "circuit_lat", "lng": "circuit_lng", "alt": "circuit_alt"}
    )
    circuits = _safe_numeric(
        circuits, ["circuitId", "circuit_lat", "circuit_lng", "circuit_alt"]
    )
    circuits = _add_circuit_profile_features(circuits).drop(columns=["circuitRef"])

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


def _recent_boolean_rate(values: deque[float], window: int) -> float:
    recent = list(values)[-window:]
    if not recent:
        return np.nan
    return float(np.mean(recent))


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


def _recent_outcome_features(
    dnf_history: deque[float],
    qualy_delta_history: deque[float],
    prefix: str,
) -> dict[str, float]:
    features = {}
    for window in (3, 5, 10):
        features[f"{prefix}_last_{window}_dnf_rate"] = _recent_boolean_rate(
            dnf_history, window
        )
        features[f"{prefix}_last_{window}_qualy_finish_delta"] = _recent_average(
            qualy_delta_history, window
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


def _recent_constructor_pair_features(
    avg_finish_history: deque[float],
    best_finish_history: deque[float],
    points_history: deque[float],
    both_top10_history: deque[float],
    double_dnf_history: deque[float],
) -> dict[str, float]:
    features = {}
    for window in (3, 5):
        features[f"constructor_pair_last_{window}_avg_finish_position"] = (
            _recent_average(avg_finish_history, window)
        )
        features[f"constructor_pair_last_{window}_best_finish_position"] = (
            _recent_average(best_finish_history, window)
        )
        features[f"constructor_pair_last_{window}_points"] = _recent_average(
            points_history, window
        )
        features[f"constructor_pair_last_{window}_both_top10_rate"] = (
            _recent_boolean_rate(both_top10_history, window)
        )
        features[f"constructor_pair_last_{window}_double_dnf_rate"] = (
            _recent_boolean_rate(double_dnf_history, window)
        )
    return features


def _rate_from_stats(stats: RunningStats) -> float:
    return stats.mean


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
    driver_circuit_dnf_stats: dict[tuple[int, int], RunningStats] = {}
    driver_similar_circuit_finish_stats: dict[tuple[int, tuple[int, int, int]], RunningStats] = {}
    driver_similar_circuit_top10_stats: dict[tuple[int, tuple[int, int, int]], RunningStats] = {}
    driver_similar_circuit_dnf_stats: dict[tuple[int, tuple[int, int, int]], RunningStats] = {}
    constructor_position_stats: dict[int, RunningStats] = {}
    constructor_circuit_finish_stats: dict[tuple[int, int], RunningStats] = {}
    constructor_circuit_top10_stats: dict[tuple[int, int], RunningStats] = {}
    constructor_circuit_dnf_stats: dict[tuple[int, int], RunningStats] = {}
    constructor_similar_circuit_finish_stats: dict[tuple[int, tuple[int, int, int]], RunningStats] = {}
    constructor_similar_circuit_top10_stats: dict[tuple[int, tuple[int, int, int]], RunningStats] = {}
    constructor_similar_circuit_dnf_stats: dict[tuple[int, tuple[int, int, int]], RunningStats] = {}
    constructor_speed_finish_stats: dict[tuple[int, int], RunningStats] = {}
    constructor_speed_top10_stats: dict[tuple[int, int], RunningStats] = {}
    constructor_speed_dnf_stats: dict[tuple[int, int], RunningStats] = {}
    constructor_aero_finish_stats: dict[tuple[int, int], RunningStats] = {}
    constructor_aero_top10_stats: dict[tuple[int, int], RunningStats] = {}
    constructor_aero_dnf_stats: dict[tuple[int, int], RunningStats] = {}
    constructor_street_finish_stats: dict[tuple[int, int], RunningStats] = {}
    constructor_street_top10_stats: dict[tuple[int, int], RunningStats] = {}
    constructor_street_dnf_stats: dict[tuple[int, int], RunningStats] = {}

    driver_latest_points: dict[int, float] = {}
    driver_latest_wins: dict[int, float] = {}
    constructor_latest_points: dict[int, float] = {}
    constructor_latest_wins: dict[int, float] = {}
    driver_circuit_wins: dict[tuple[int, int], int] = {}
    constructor_circuit_wins: dict[tuple[int, int], int] = {}
    driver_recent_finishes: dict[int, deque[float]] = {}
    driver_recent_dnfs: dict[int, deque[float]] = {}
    driver_recent_qualy_deltas: dict[int, deque[float]] = {}
    constructor_recent_finishes: dict[int, deque[float]] = {}
    constructor_recent_dnfs: dict[int, deque[float]] = {}
    constructor_recent_qualy_deltas: dict[int, deque[float]] = {}
    constructor_pair_avg_finishes: dict[int, deque[float]] = {}
    constructor_pair_best_finishes: dict[int, deque[float]] = {}
    constructor_pair_points: dict[int, deque[float]] = {}
    constructor_pair_both_top10: dict[int, deque[float]] = {}
    constructor_pair_double_dnf: dict[int, deque[float]] = {}
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
            circuit_profile_key = _circuit_profile_key(row)
            driver_similar_circuit_key = (driver_id, circuit_profile_key)
            constructor_similar_circuit_key = (constructor_id, circuit_profile_key)
            constructor_speed_key = (
                constructor_id,
                _profile_bucket(row.circuit_speed_score),
            )
            constructor_aero_key = (
                constructor_id,
                _profile_bucket(row.circuit_aero_load_score),
            )
            constructor_street_key = (
                constructor_id,
                _profile_bucket(row.circuit_is_street),
            )
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
            driver_circuit_dnf = driver_circuit_dnf_stats.get(
                driver_circuit_key, RunningStats()
            )
            driver_similar_circuit_finish = driver_similar_circuit_finish_stats.get(
                driver_similar_circuit_key, RunningStats()
            )
            driver_similar_circuit_top10 = driver_similar_circuit_top10_stats.get(
                driver_similar_circuit_key, RunningStats()
            )
            driver_similar_circuit_dnf = driver_similar_circuit_dnf_stats.get(
                driver_similar_circuit_key, RunningStats()
            )
            constructor_pos = constructor_position_stats.get(
                constructor_id, RunningStats()
            )
            constructor_circuit_finish = constructor_circuit_finish_stats.get(
                constructor_circuit_key, RunningStats()
            )
            constructor_circuit_top10 = constructor_circuit_top10_stats.get(
                constructor_circuit_key, RunningStats()
            )
            constructor_circuit_dnf = constructor_circuit_dnf_stats.get(
                constructor_circuit_key, RunningStats()
            )
            constructor_similar_circuit_finish = constructor_similar_circuit_finish_stats.get(
                constructor_similar_circuit_key, RunningStats()
            )
            constructor_similar_circuit_top10 = constructor_similar_circuit_top10_stats.get(
                constructor_similar_circuit_key, RunningStats()
            )
            constructor_similar_circuit_dnf = constructor_similar_circuit_dnf_stats.get(
                constructor_similar_circuit_key, RunningStats()
            )
            constructor_speed_finish = constructor_speed_finish_stats.get(
                constructor_speed_key, RunningStats()
            )
            constructor_speed_top10 = constructor_speed_top10_stats.get(
                constructor_speed_key, RunningStats()
            )
            constructor_speed_dnf = constructor_speed_dnf_stats.get(
                constructor_speed_key, RunningStats()
            )
            constructor_aero_finish = constructor_aero_finish_stats.get(
                constructor_aero_key, RunningStats()
            )
            constructor_aero_top10 = constructor_aero_top10_stats.get(
                constructor_aero_key, RunningStats()
            )
            constructor_aero_dnf = constructor_aero_dnf_stats.get(
                constructor_aero_key, RunningStats()
            )
            constructor_street_finish = constructor_street_finish_stats.get(
                constructor_street_key, RunningStats()
            )
            constructor_street_top10 = constructor_street_top10_stats.get(
                constructor_street_key, RunningStats()
            )
            constructor_street_dnf = constructor_street_dnf_stats.get(
                constructor_street_key, RunningStats()
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
                    "driver_circuit_dnf_rate_hist": _rate_from_stats(driver_circuit_dnf),
                    "driver_similar_circuit_finish_position_mean_hist": (
                        driver_similar_circuit_finish.mean
                    ),
                    "driver_similar_circuit_top10_rate_hist": _rate_from_stats(
                        driver_similar_circuit_top10
                    ),
                    "driver_similar_circuit_dnf_rate_hist": _rate_from_stats(
                        driver_similar_circuit_dnf
                    ),
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
                    "constructor_circuit_top10_rate_hist": _rate_from_stats(
                        constructor_circuit_top10
                    ),
                    "constructor_circuit_dnf_rate_hist": _rate_from_stats(
                        constructor_circuit_dnf
                    ),
                    "constructor_season_points_before_race": constructor_season_points.get(
                        constructor_season_key, 0.0
                    ),
                    "constructor_similar_circuit_finish_position_mean_hist": (
                        constructor_similar_circuit_finish.mean
                    ),
                    "constructor_similar_circuit_top10_rate_hist": _rate_from_stats(
                        constructor_similar_circuit_top10
                    ),
                    "constructor_similar_circuit_dnf_rate_hist": _rate_from_stats(
                        constructor_similar_circuit_dnf
                    ),
                    "constructor_speed_profile_finish_position_mean_hist": (
                        constructor_speed_finish.mean
                    ),
                    "constructor_speed_profile_top10_rate_hist": _rate_from_stats(
                        constructor_speed_top10
                    ),
                    "constructor_speed_profile_dnf_rate_hist": _rate_from_stats(
                        constructor_speed_dnf
                    ),
                    "constructor_aero_profile_finish_position_mean_hist": (
                        constructor_aero_finish.mean
                    ),
                    "constructor_aero_profile_top10_rate_hist": _rate_from_stats(
                        constructor_aero_top10
                    ),
                    "constructor_aero_profile_dnf_rate_hist": _rate_from_stats(
                        constructor_aero_dnf
                    ),
                    "constructor_street_finish_position_mean_hist": (
                        constructor_street_finish.mean
                    ),
                    "constructor_street_top10_rate_hist": _rate_from_stats(
                        constructor_street_top10
                    ),
                    "constructor_street_dnf_rate_hist": _rate_from_stats(
                        constructor_street_dnf
                    ),
                }
            row_features.update(
                _recent_driver_features(
                    driver_recent_finishes.get(driver_id, deque(maxlen=10)),
                    "driver",
                )
            )
            row_features.update(
                _recent_outcome_features(
                    driver_recent_dnfs.get(driver_id, deque(maxlen=10)),
                    driver_recent_qualy_deltas.get(driver_id, deque(maxlen=10)),
                    "driver",
                )
            )
            row_features.update(
                _recent_constructor_features(
                    constructor_recent_finishes.get(constructor_id, deque(maxlen=20))
                )
            )
            row_features.update(
                _recent_outcome_features(
                    constructor_recent_dnfs.get(constructor_id, deque(maxlen=20)),
                    constructor_recent_qualy_deltas.get(
                        constructor_id, deque(maxlen=20)
                    ),
                    "constructor",
                )
            )
            row_features.update(
                _recent_constructor_pair_features(
                    constructor_pair_avg_finishes.get(constructor_id, deque(maxlen=5)),
                    constructor_pair_best_finishes.get(constructor_id, deque(maxlen=5)),
                    constructor_pair_points.get(constructor_id, deque(maxlen=5)),
                    constructor_pair_both_top10.get(constructor_id, deque(maxlen=5)),
                    constructor_pair_double_dnf.get(constructor_id, deque(maxlen=5)),
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
            circuit_profile_key = _circuit_profile_key(row)
            driver_similar_key = (driver_id, circuit_profile_key)
            constructor_similar_key = (constructor_id, circuit_profile_key)
            constructor_speed_key = (
                constructor_id,
                _profile_bucket(row.circuit_speed_score),
            )
            constructor_aero_key = (
                constructor_id,
                _profile_bucket(row.circuit_aero_load_score),
            )
            constructor_street_key = (
                constructor_id,
                _profile_bucket(row.circuit_is_street),
            )
            dnf_value = 0.0 if _is_classified_finish(row.statusId) else 1.0
            top10_value = 1.0 if row.positionOrder <= 10 else 0.0
            qualy_delta = _qualy_finish_delta(row.positionOrder, row.qualifying_position)
            driver_circuit_finish_stats.setdefault(driver_key, RunningStats()).add(
                row.positionOrder
            )
            driver_circuit_dnf_stats.setdefault(driver_key, RunningStats()).add(
                dnf_value
            )
            driver_similar_circuit_finish_stats.setdefault(
                driver_similar_key, RunningStats()
            ).add(row.positionOrder)
            driver_similar_circuit_top10_stats.setdefault(
                driver_similar_key, RunningStats()
            ).add(top10_value)
            driver_similar_circuit_dnf_stats.setdefault(
                driver_similar_key, RunningStats()
            ).add(dnf_value)
            constructor_circuit_finish_stats.setdefault(constructor_key, RunningStats()).add(
                row.positionOrder
            )
            constructor_circuit_top10_stats.setdefault(
                constructor_key, RunningStats()
            ).add(top10_value)
            constructor_circuit_dnf_stats.setdefault(
                constructor_key, RunningStats()
            ).add(dnf_value)
            constructor_similar_circuit_finish_stats.setdefault(
                constructor_similar_key, RunningStats()
            ).add(row.positionOrder)
            constructor_similar_circuit_top10_stats.setdefault(
                constructor_similar_key, RunningStats()
            ).add(top10_value)
            constructor_similar_circuit_dnf_stats.setdefault(
                constructor_similar_key, RunningStats()
            ).add(dnf_value)
            constructor_speed_finish_stats.setdefault(
                constructor_speed_key, RunningStats()
            ).add(row.positionOrder)
            constructor_speed_top10_stats.setdefault(
                constructor_speed_key, RunningStats()
            ).add(top10_value)
            constructor_speed_dnf_stats.setdefault(
                constructor_speed_key, RunningStats()
            ).add(dnf_value)
            constructor_aero_finish_stats.setdefault(
                constructor_aero_key, RunningStats()
            ).add(row.positionOrder)
            constructor_aero_top10_stats.setdefault(
                constructor_aero_key, RunningStats()
            ).add(top10_value)
            constructor_aero_dnf_stats.setdefault(
                constructor_aero_key, RunningStats()
            ).add(dnf_value)
            constructor_street_finish_stats.setdefault(
                constructor_street_key, RunningStats()
            ).add(row.positionOrder)
            constructor_street_top10_stats.setdefault(
                constructor_street_key, RunningStats()
            ).add(top10_value)
            constructor_street_dnf_stats.setdefault(
                constructor_street_key, RunningStats()
            ).add(dnf_value)
            if row.positionOrder == 1:
                driver_circuit_wins[driver_key] = driver_circuit_wins.get(driver_key, 0) + 1
                constructor_circuit_wins[constructor_key] = (
                    constructor_circuit_wins.get(constructor_key, 0) + 1
                )
            driver_recent_finishes.setdefault(driver_id, deque(maxlen=10)).append(
                row.positionOrder
            )
            driver_recent_dnfs.setdefault(driver_id, deque(maxlen=10)).append(dnf_value)
            driver_recent_qualy_deltas.setdefault(driver_id, deque(maxlen=10)).append(
                qualy_delta
            )
            constructor_recent_finishes.setdefault(
                constructor_id, deque(maxlen=20)
            ).append(row.positionOrder)
            constructor_recent_dnfs.setdefault(
                constructor_id, deque(maxlen=20)
            ).append(dnf_value)
            constructor_recent_qualy_deltas.setdefault(
                constructor_id, deque(maxlen=20)
            ).append(qualy_delta)
            driver_season_key = (driver_id, year)
            constructor_season_key = (constructor_id, year)
            points = 0.0 if pd.isna(row.points) else float(row.points)
            driver_season_points[driver_season_key] = (
                driver_season_points.get(driver_season_key, 0.0) + points
            )
            constructor_season_points[constructor_season_key] = (
                constructor_season_points.get(constructor_season_key, 0.0) + points
            )

        for constructor_id, constructor_rows in race_rows.groupby("constructorId"):
            constructor_id = int(constructor_id)
            finish_positions = constructor_rows["positionOrder"].astype(float)
            points = constructor_rows["points"].fillna(0.0).astype(float)
            dnf_values = constructor_rows["statusId"].map(
                lambda value: 0.0 if _is_classified_finish(value) else 1.0
            )

            constructor_pair_avg_finishes.setdefault(
                constructor_id, deque(maxlen=5)
            ).append(float(finish_positions.mean()))
            constructor_pair_best_finishes.setdefault(
                constructor_id, deque(maxlen=5)
            ).append(float(finish_positions.min()))
            constructor_pair_points.setdefault(constructor_id, deque(maxlen=5)).append(
                float(points.sum())
            )
            constructor_pair_both_top10.setdefault(
                constructor_id, deque(maxlen=5)
            ).append(float((finish_positions <= 10).all()))
            constructor_pair_double_dnf.setdefault(
                constructor_id, deque(maxlen=5)
            ).append(float((dnf_values == 1.0).all()))

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
