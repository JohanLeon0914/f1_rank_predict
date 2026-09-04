from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "red_win"
DATE_COLUMN = "event_date"
ID_COLUMNS = ["fight_id", "event_id", "event_date", "red_fighter_id", "blue_fighter_id"]
WINDOWS = (3, 5, 10)

PROFILE_NUMERIC_COLUMNS = [
    "height_inches",
    "weight_lbs",
    "reach_inches",
    "age_years",
]

HISTORY_BASE_METRICS = [
    "win",
    "ko_tko_win",
    "submission_win",
    "decision_win",
    "ko_tko_loss",
    "submission_loss",
    "decision_loss",
    "finish_win",
    "finish_loss",
    "kd_for",
    "kd_against",
    "sig_landed_for",
    "sig_attempted_for",
    "sig_landed_against",
    "sig_attempted_against",
    "total_str_landed_for",
    "total_str_attempted_for",
    "total_str_landed_against",
    "total_str_attempted_against",
    "td_success_for",
    "td_attempted_for",
    "td_success_against",
    "td_attempted_against",
    "sub_att_for",
    "sub_att_against",
    "rev_for",
    "rev_against",
    "ctrl_seconds_for",
    "ctrl_seconds_against",
    "rounds_fought",
    "fight_duration_seconds",
]

RATE_FEATURES = {
    "sig_acc_for": ("sig_landed_for", "sig_attempted_for"),
    "sig_acc_against": ("sig_landed_against", "sig_attempted_against"),
    "td_acc_for": ("td_success_for", "td_attempted_for"),
    "td_acc_against": ("td_success_against", "td_attempted_against"),
}

METHOD_KO_TKO = ("KO/TKO", "TKO", "DQ")
METHOD_SUBMISSION = ("Submission",)
METHOD_DECISION = ("Decision",)


@dataclass
class FighterHistory:
    fights: int = 0
    wins: int = 0
    losses: int = 0
    methods_for: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    methods_against: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    recent: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=10))
    weight_classes: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, fight_stats: dict[str, float], weight_class: str, won: bool, method_group: str) -> None:
        self.fights += 1
        self.wins += int(won)
        self.losses += int(not won)
        if won:
            self.methods_for[method_group] += 1
        else:
            self.methods_against[method_group] += 1
        self.weight_classes[str(weight_class)] += 1
        for key, value in fight_stats.items():
            if pd.notna(value):
                self.totals[key] += float(value)
        self.recent.append({**fight_stats, "win": float(won), f"{method_group}_win": float(won), f"{method_group}_loss": float(not won)})


def parse_height_to_inches(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace('"', "").strip()
    if "'" not in text:
        return np.nan
    feet, inches = text.split("'", maxsplit=1)
    try:
        return int(feet.strip()) * 12 + float(inches.strip() or 0)
    except ValueError:
        return np.nan


def parse_time_to_seconds(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value)
    if ":" not in text:
        return np.nan
    minutes, seconds = text.split(":", maxsplit=1)
    try:
        return int(minutes) * 60 + float(seconds)
    except ValueError:
        return np.nan


def method_group(method: Any) -> str:
    text = "" if pd.isna(method) else str(method)
    if any(token in text for token in METHOD_KO_TKO):
        return "ko_tko"
    if any(token in text for token in METHOD_SUBMISSION):
        return "submission"
    if any(token in text for token in METHOD_DECISION):
        return "decision"
    return "other"


def normalize_fighters(fighters: pd.DataFrame) -> pd.DataFrame:
    output = fighters.copy()
    output["height_inches"] = output["height"].map(parse_height_to_inches)
    output["dob"] = pd.to_datetime(output["dob"], errors="coerce")
    output["stance"] = output["stance"].fillna("Unknown").astype(str)
    numeric = [
        "weight_lbs",
        "reach_inches",
        "slpm",
        "str_acc",
        "sapm",
        "str_def",
        "td_avg",
        "td_acc",
        "td_def",
        "sub_avg",
    ]
    for column in numeric:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.set_index("fighter_id", drop=False)


def prepare_fights(master: pd.DataFrame) -> pd.DataFrame:
    output = master.copy()
    output["event_date"] = pd.to_datetime(output["event_date"], errors="coerce")
    output = output.dropna(subset=["event_date", "r_fighter_id", "b_fighter_id", "winner_id"])
    output = output[output["result_status"].eq("win")]
    output = output[
        (output["winner_id"].eq(output["r_fighter_id"]))
        | (output["winner_id"].eq(output["b_fighter_id"]))
    ].copy()
    output["red_win"] = output["winner_id"].eq(output["r_fighter_id"]).astype(int)
    output["method_group"] = output["method"].map(method_group)
    output["finish_time_seconds"] = output["finish_time"].map(parse_time_to_seconds)
    output["fight_duration_seconds"] = (
        (pd.to_numeric(output["finish_round"], errors="coerce") - 1).clip(lower=0) * 300
        + output["finish_time_seconds"].fillna(0)
    )
    output["rounds_fought"] = pd.to_numeric(output["rounds_fought"], errors="coerce")
    return output.sort_values(["event_date", "fight_id"]).reset_index(drop=True)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def _side_totals(row: pd.Series, side: str, opponent_side: str) -> dict[str, float]:
    prefix = f"{side}_total_"
    opponent_prefix = f"{opponent_side}_total_"
    return {
        "kd_for": row.get(f"{prefix}kd", np.nan),
        "kd_against": row.get(f"{opponent_prefix}kd", np.nan),
        "sig_landed_for": row.get(f"{prefix}sig_landed", np.nan),
        "sig_attempted_for": row.get(f"{prefix}sig_atmp", np.nan),
        "sig_landed_against": row.get(f"{opponent_prefix}sig_landed", np.nan),
        "sig_attempted_against": row.get(f"{opponent_prefix}sig_atmp", np.nan),
        "total_str_landed_for": row.get(f"{prefix}total_str_landed", np.nan),
        "total_str_attempted_for": row.get(f"{prefix}total_str_atmp", np.nan),
        "total_str_landed_against": row.get(f"{opponent_prefix}total_str_landed", np.nan),
        "total_str_attempted_against": row.get(f"{opponent_prefix}total_str_atmp", np.nan),
        "td_success_for": row.get(f"{prefix}td_success", np.nan),
        "td_attempted_for": row.get(f"{prefix}td_atmp", np.nan),
        "td_success_against": row.get(f"{opponent_prefix}td_success", np.nan),
        "td_attempted_against": row.get(f"{opponent_prefix}td_atmp", np.nan),
        "sub_att_for": row.get(f"{prefix}sub_att", np.nan),
        "sub_att_against": row.get(f"{opponent_prefix}sub_att", np.nan),
        "rev_for": row.get(f"{prefix}rev", np.nan),
        "rev_against": row.get(f"{opponent_prefix}rev", np.nan),
        "ctrl_seconds_for": row.get(f"{prefix}ctrl_seconds", np.nan),
        "ctrl_seconds_against": row.get(f"{opponent_prefix}ctrl_seconds", np.nan),
        "rounds_fought": row.get("rounds_fought", np.nan),
        "fight_duration_seconds": row.get("fight_duration_seconds", np.nan),
    }


def _profile_values(
    fighters: pd.DataFrame,
    fighter_id: str,
    event_date: pd.Timestamp,
    prefix: str,
) -> dict[str, Any]:
    if fighter_id not in fighters.index:
        return {f"{prefix}_{column}": np.nan for column in PROFILE_NUMERIC_COLUMNS} | {
            f"{prefix}_stance": "Unknown",
            f"{prefix}_fighter_name": fighter_id,
        }
    row = fighters.loc[fighter_id]
    values = {f"{prefix}_{column}": row.get(column, np.nan) for column in PROFILE_NUMERIC_COLUMNS}
    dob = row.get("dob")
    values[f"{prefix}_age_years"] = (
        (event_date - dob).days / 365.25 if pd.notna(dob) else np.nan
    )
    values[f"{prefix}_stance"] = row.get("stance", "Unknown")
    values[f"{prefix}_fighter_name"] = row.get("fighter_name", fighter_id)
    return values


def _history_features(history: FighterHistory, prefix: str) -> dict[str, float]:
    features = {
        f"{prefix}_career_fights": float(history.fights),
        f"{prefix}_career_wins": float(history.wins),
        f"{prefix}_career_losses": float(history.losses),
        f"{prefix}_career_win_rate": _safe_div(history.wins, history.fights),
        f"{prefix}_career_ko_tko_win_rate": _safe_div(history.methods_for["ko_tko"], history.fights),
        f"{prefix}_career_submission_win_rate": _safe_div(history.methods_for["submission"], history.fights),
        f"{prefix}_career_decision_win_rate": _safe_div(history.methods_for["decision"], history.fights),
        f"{prefix}_career_finish_loss_rate": _safe_div(
            history.methods_against["ko_tko"] + history.methods_against["submission"],
            history.fights,
        ),
    }
    for metric in HISTORY_BASE_METRICS:
        features[f"{prefix}_career_avg_{metric}"] = _safe_div(history.totals[metric], history.fights)
    for feature_name, (num, den) in RATE_FEATURES.items():
        features[f"{prefix}_career_{feature_name}"] = _safe_div(history.totals[num], history.totals[den])

    recent_rows = list(history.recent)
    for window in WINDOWS:
        rows = recent_rows[-window:]
        features[f"{prefix}_last_{window}_fight_count"] = float(len(rows))
        for metric in HISTORY_BASE_METRICS:
            values = [row.get(metric, np.nan) for row in rows]
            clean = [value for value in values if pd.notna(value)]
            features[f"{prefix}_last_{window}_avg_{metric}"] = float(np.mean(clean)) if clean else np.nan
        for feature_name, (num, den) in RATE_FEATURES.items():
            num_total = sum(row.get(num, 0.0) for row in rows if pd.notna(row.get(num, np.nan)))
            den_total = sum(row.get(den, 0.0) for row in rows if pd.notna(row.get(den, np.nan)))
            features[f"{prefix}_last_{window}_{feature_name}"] = _safe_div(num_total, den_total)
    return features


def _add_differences(features: dict[str, Any]) -> None:
    comparable = [
        "height_inches",
        "weight_lbs",
        "reach_inches",
        "age_years",
    ]
    for suffix in comparable:
        red = features.get(f"red_{suffix}", np.nan)
        blue = features.get(f"blue_{suffix}", np.nan)
        features[f"diff_{suffix}"] = red - blue if pd.notna(red) and pd.notna(blue) else np.nan

    dynamic_suffixes = [
        key.replace("red_", "", 1)
        for key in list(features)
        if key.startswith("red_career_") or key.startswith("red_last_")
    ]
    for suffix in dynamic_suffixes:
        red = features.get(f"red_{suffix}", np.nan)
        blue = features.get(f"blue_{suffix}", np.nan)
        features[f"diff_{suffix}"] = red - blue if pd.notna(red) and pd.notna(blue) else np.nan


def _pair_key(red_id: str, blue_id: str) -> tuple[str, str]:
    return tuple(sorted((str(red_id), str(blue_id))))


def _head_to_head_features(
    head_to_head: dict[tuple[str, str], dict[str, int]],
    red_id: str,
    blue_id: str,
) -> dict[str, float]:
    record = head_to_head[_pair_key(red_id, blue_id)]
    red_wins = record.get(str(red_id), 0)
    blue_wins = record.get(str(blue_id), 0)
    total = red_wins + blue_wins
    return {
        "h2h_total_fights": float(total),
        "h2h_red_wins": float(red_wins),
        "h2h_blue_wins": float(blue_wins),
        "h2h_red_win_rate": _safe_div(red_wins, total),
        "h2h_win_diff": float(red_wins - blue_wins),
    }


def build_model_dataset(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fighters = normalize_fighters(datasets["fighter"])
    fights = prepare_fights(datasets["master"])
    histories: dict[str, FighterHistory] = defaultdict(FighterHistory)
    head_to_head: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rows: list[dict[str, Any]] = []

    for fight in fights.itertuples(index=False):
        row = pd.Series(fight._asdict())
        red_id = str(row["r_fighter_id"])
        blue_id = str(row["b_fighter_id"])
        event_date = row["event_date"]
        features: dict[str, Any] = {
            "fight_id": row["fight_id"],
            "event_id": row["event_id"],
            "event_date": event_date,
            "red_fighter_id": red_id,
            "blue_fighter_id": blue_id,
            "weight_class": row.get("weight_class", "Unknown"),
            "title_fight": int(row.get("title_fight", 0) or 0),
            "red_win": int(row["red_win"]),
        }
        features.update(_profile_values(fighters, red_id, event_date, "red"))
        features.update(_profile_values(fighters, blue_id, event_date, "blue"))
        features["stance_matchup"] = f"{features['red_stance']}_vs_{features['blue_stance']}"
        features.update(_history_features(histories[red_id], "red"))
        features.update(_history_features(histories[blue_id], "blue"))
        features.update(_head_to_head_features(head_to_head, red_id, blue_id))
        _add_differences(features)
        rows.append(features)

        red_won = bool(row["red_win"])
        group = str(row["method_group"])
        red_stats = _side_totals(row, "r", "b")
        blue_stats = _side_totals(row, "b", "r")
        red_stats.update(
            {
                "win": float(red_won),
                "ko_tko_win": float(red_won and group == "ko_tko"),
                "submission_win": float(red_won and group == "submission"),
                "decision_win": float(red_won and group == "decision"),
                "finish_win": float(red_won and group in {"ko_tko", "submission"}),
                "ko_tko_loss": float((not red_won) and group == "ko_tko"),
                "submission_loss": float((not red_won) and group == "submission"),
                "decision_loss": float((not red_won) and group == "decision"),
                "finish_loss": float((not red_won) and group in {"ko_tko", "submission"}),
            }
        )
        blue_stats.update(
            {
                "win": float(not red_won),
                "ko_tko_win": float((not red_won) and group == "ko_tko"),
                "submission_win": float((not red_won) and group == "submission"),
                "decision_win": float((not red_won) and group == "decision"),
                "finish_win": float((not red_won) and group in {"ko_tko", "submission"}),
                "ko_tko_loss": float(red_won and group == "ko_tko"),
                "submission_loss": float(red_won and group == "submission"),
                "decision_loss": float(red_won and group == "decision"),
                "finish_loss": float(red_won and group in {"ko_tko", "submission"}),
            }
        )
        histories[red_id].add(red_stats, row.get("weight_class", "Unknown"), red_won, group)
        histories[blue_id].add(blue_stats, row.get("weight_class", "Unknown"), not red_won, group)
        head_to_head[_pair_key(red_id, blue_id)][row["winner_id"]] += 1

    dataset = pd.DataFrame(rows)
    dataset["event_date"] = pd.to_datetime(dataset["event_date"], errors="coerce")
    return dataset


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLUMNS + [TARGET_COLUMN, "red_fighter_name", "blue_fighter_name"])
    return [column for column in dataset.columns if column not in excluded]


def categorical_columns(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if column in {"weight_class", "red_stance", "blue_stance", "stance_matchup"}
    ]


def numeric_columns(columns: list[str]) -> list[str]:
    categorical = set(categorical_columns(columns))
    return [column for column in columns if column not in categorical]


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    numeric = numeric_columns(columns)
    categorical = categorical_columns(columns)
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
