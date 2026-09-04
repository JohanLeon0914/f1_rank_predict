from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from ufc_predictor.config import UFC_MODEL_DIR
from ufc_predictor.data_loading import load_ufc_data
from ufc_predictor.feature_engineering import (
    FighterHistory,
    _add_differences,
    _head_to_head_features,
    _history_features,
    _pair_key,
    _profile_values,
    _side_totals,
    method_group,
    normalize_fighters,
    prepare_fights,
)
from ufc_predictor.training import load_trained_ufc_model


EXPLANATION_GROUPS = {
    "fighter_profile": [
        "height_inches",
        "weight_lbs",
        "reach_inches",
        "age_years",
        "stance",
    ],
    "career_summary": [
        "career_fights",
        "career_wins",
        "career_losses",
        "career_win_rate",
        "career_ko_tko_win_rate",
        "career_submission_win_rate",
        "career_decision_win_rate",
        "career_finish_loss_rate",
        "career_sig_acc_for",
        "career_sig_acc_against",
        "career_td_acc_for",
        "career_td_acc_against",
    ],
    "last_3": [
        "last_3_fight_count",
        "last_3_avg_win",
        "last_3_avg_kd_for",
        "last_3_avg_kd_against",
        "last_3_avg_sig_landed_for",
        "last_3_avg_sig_landed_against",
        "last_3_sig_acc_for",
        "last_3_td_acc_for",
        "last_3_avg_ctrl_seconds_for",
    ],
    "last_5": [
        "last_5_fight_count",
        "last_5_avg_win",
        "last_5_avg_kd_for",
        "last_5_avg_kd_against",
        "last_5_avg_sig_landed_for",
        "last_5_avg_sig_landed_against",
        "last_5_sig_acc_for",
        "last_5_td_acc_for",
        "last_5_avg_ctrl_seconds_for",
    ],
    "last_10": [
        "last_10_fight_count",
        "last_10_avg_win",
        "last_10_avg_kd_for",
        "last_10_avg_kd_against",
        "last_10_avg_sig_landed_for",
        "last_10_avg_sig_landed_against",
        "last_10_sig_acc_for",
        "last_10_td_acc_for",
        "last_10_avg_ctrl_seconds_for",
    ],
}


def _safe_value(value: Any) -> Any:
    if isinstance(value, float) and pd.isna(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _load_feature_importance(model_dir: Path) -> list[dict[str, Any]]:
    path = model_dir / "ufc_feature_importance.csv"
    if not path.exists():
        return []
    report = pd.read_csv(path)
    return [
        {
            "feature": str(row.original_feature),
            "importance": float(row.importance),
            "importance_pct": float(row.importance_pct),
        }
        for row in report.head(25).itertuples()
    ]


def _load_metrics(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "ufc_metrics.joblib"
    if not path.exists():
        return {}
    return joblib.load(path)


def _clean_transformed_feature_name(name: str) -> str:
    if name.startswith("numeric__"):
        return name.replace("numeric__", "", 1)
    if name.startswith("categorical__"):
        return name.replace("categorical__", "", 1)
    return name


def _aggregate_contributions(
    transformed_feature_names: np.ndarray,
    contribution_values: np.ndarray,
) -> list[dict[str, Any]]:
    rows = pd.DataFrame(
        {
            "feature": [_clean_transformed_feature_name(name) for name in transformed_feature_names],
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
            "direction": "red" if row.contribution >= 0 else "blue",
        }
        for row in grouped.head(15).itertuples()
    ]


def _history_snapshot(features: dict[str, Any], prefix: str) -> dict[str, dict[str, Any]]:
    output = {}
    for group, suffixes in EXPLANATION_GROUPS.items():
        group_values = {}
        for suffix in suffixes:
            key = f"{prefix}_{suffix}"
            if key in features:
                group_values[suffix] = _safe_value(features[key])
        output[group] = group_values
    return output


def _recent_fights(fights: pd.DataFrame, fighter_id: str, before_date: pd.Timestamp, limit: int = 10) -> list[dict[str, Any]]:
    rows = fights[
        (fights["event_date"] < before_date)
        & ((fights["r_fighter_id"].eq(fighter_id)) | (fights["b_fighter_id"].eq(fighter_id)))
    ].sort_values("event_date", ascending=False)
    output = []
    for row in rows.head(limit).itertuples():
        is_red = row.r_fighter_id == fighter_id
        opponent_id = row.b_fighter_id if is_red else row.r_fighter_id
        won = row.winner_id == fighter_id
        output.append(
            {
                "fight_id": row.fight_id,
                "event_date": _safe_value(row.event_date),
                "event_name": getattr(row, "event_name", None),
                "opponent_id": opponent_id,
                "result": "win" if won else "loss",
                "method": row.method,
                "weight_class": row.weight_class,
            }
        )
    return output


def _update_histories_until(
    fights: pd.DataFrame,
    target_date: pd.Timestamp,
) -> tuple[dict[str, FighterHistory], dict[tuple[str, str], dict[str, int]]]:
    from collections import defaultdict

    histories: dict[str, FighterHistory] = defaultdict(FighterHistory)
    head_to_head: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    historical = fights[fights["event_date"] < target_date].copy()
    for fight in historical.sort_values(["event_date", "fight_id"]).itertuples(index=False):
        row = pd.Series(fight._asdict())
        red_id = str(row["r_fighter_id"])
        blue_id = str(row["b_fighter_id"])
        red_won = bool(row["red_win"])
        group = str(row["method_group"])
        red_stats = _side_totals(row, "r", "b")
        blue_stats = _side_totals(row, "b", "r")
        red_stats.update({"win": float(red_won)})
        blue_stats.update({"win": float(not red_won)})
        for outcome in ("ko_tko", "submission", "decision"):
            red_stats[f"{outcome}_win"] = float(red_won and group == outcome)
            red_stats[f"{outcome}_loss"] = float((not red_won) and group == outcome)
            blue_stats[f"{outcome}_win"] = float((not red_won) and group == outcome)
            blue_stats[f"{outcome}_loss"] = float(red_won and group == outcome)
        red_stats["finish_win"] = float(red_won and group in {"ko_tko", "submission"})
        red_stats["finish_loss"] = float((not red_won) and group in {"ko_tko", "submission"})
        blue_stats["finish_win"] = float((not red_won) and group in {"ko_tko", "submission"})
        blue_stats["finish_loss"] = float(red_won and group in {"ko_tko", "submission"})
        histories[red_id].add(red_stats, row.get("weight_class", "Unknown"), red_won, group)
        histories[blue_id].add(blue_stats, row.get("weight_class", "Unknown"), not red_won, group)
        head_to_head[_pair_key(red_id, blue_id)][str(row["winner_id"])] += 1
    return histories, head_to_head


def _infer_weight_class(fights: pd.DataFrame, red_id: str, blue_id: str, before_date: pd.Timestamp) -> str:
    previous = fights[
        (fights["event_date"] < before_date)
        & (
            fights["r_fighter_id"].isin([red_id, blue_id])
            | fights["b_fighter_id"].isin([red_id, blue_id])
        )
    ].sort_values("event_date", ascending=False)
    if previous.empty:
        return "Unknown"
    return str(previous["weight_class"].mode().iloc[0])


def build_inference_features(
    red_fighter_id: str,
    blue_fighter_id: str,
    fight_date: str | None = None,
    weight_class: str | None = None,
    title_fight: bool = False,
    datasets: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    datasets = load_ufc_data() if datasets is None else datasets
    fighters = normalize_fighters(datasets["fighter"])
    fights = prepare_fights(datasets["master"])
    target_date = pd.to_datetime(fight_date) if fight_date else fights["event_date"].max() + pd.Timedelta(days=1)
    red_id = str(red_fighter_id)
    blue_id = str(blue_fighter_id)
    if red_id == blue_id:
        raise ValueError("red_fighter_id y blue_fighter_id deben ser diferentes.")
    if red_id not in fighters.index:
        raise ValueError(f"No existe red_fighter_id en fighter.csv: {red_id}")
    if blue_id not in fighters.index:
        raise ValueError(f"No existe blue_fighter_id en fighter.csv: {blue_id}")

    histories, head_to_head = _update_histories_until(fights, target_date)
    resolved_weight_class = weight_class or _infer_weight_class(fights, red_id, blue_id, target_date)
    features: dict[str, Any] = {
        "fight_id": "inference",
        "event_id": "inference",
        "event_date": target_date,
        "red_fighter_id": red_id,
        "blue_fighter_id": blue_id,
        "weight_class": resolved_weight_class,
        "title_fight": int(title_fight),
    }
    features.update(_profile_values(fighters, red_id, target_date, "red"))
    features.update(_profile_values(fighters, blue_id, target_date, "blue"))
    features["stance_matchup"] = f"{features['red_stance']}_vs_{features['blue_stance']}"
    features.update(_history_features(histories[red_id], "red"))
    features.update(_history_features(histories[blue_id], "blue"))
    features.update(_head_to_head_features(head_to_head, red_id, blue_id))
    _add_differences(features)

    context = {
        "fight_date": _safe_value(target_date),
        "weight_class": resolved_weight_class,
        "red_fighter": {
            "fighter_id": red_id,
            "name": features.get("red_fighter_name"),
            "history": _history_snapshot(features, "red"),
            "recent_fights": _recent_fights(fights, red_id, target_date, 10),
        },
        "blue_fighter": {
            "fighter_id": blue_id,
            "name": features.get("blue_fighter_name"),
            "history": _history_snapshot(features, "blue"),
            "recent_fights": _recent_fights(fights, blue_id, target_date, 10),
        },
        "head_to_head": {
            key: _safe_value(value)
            for key, value in _head_to_head_features(head_to_head, red_id, blue_id).items()
        },
    }
    return pd.DataFrame([features]), context


def predict_fight_winner(
    red_fighter_id: str,
    blue_fighter_id: str,
    fight_date: str | None = None,
    weight_class: str | None = None,
    title_fight: bool = False,
    model_dir: Path = UFC_MODEL_DIR,
) -> dict[str, Any]:
    model, artifacts = load_trained_ufc_model(model_dir)
    dataset, context = build_inference_features(
        red_fighter_id=red_fighter_id,
        blue_fighter_id=blue_fighter_id,
        fight_date=fight_date,
        weight_class=weight_class,
        title_fight=title_fight,
    )
    columns = artifacts["feature_columns"]
    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"No se pudieron construir features requeridas: {missing[:20]}")
    preprocessor = artifacts["preprocessor"]
    x_matrix = preprocessor.transform(dataset[columns])
    red_probability = float(model.predict_proba(x_matrix)[0, 1])
    blue_probability = 1.0 - red_probability

    transformed_feature_names = preprocessor.get_feature_names_out()
    contributions = model.get_booster().predict(
        xgb.DMatrix(x_matrix, feature_names=list(transformed_feature_names)),
        pred_contribs=True,
    )[0]
    winner_side = "red" if red_probability >= blue_probability else "blue"
    winner = context[f"{winner_side}_fighter"]

    return {
        "prediction": {
            "winner_side": winner_side,
            "winner_fighter_id": winner["fighter_id"],
            "winner_name": winner["name"],
            "red_win_probability": red_probability,
            "blue_win_probability": blue_probability,
            "confidence_pct": max(red_probability, blue_probability) * 100,
            "model_output_note": "Probabilities are model estimates, not guarantees or betting advice.",
        },
        "fight": {
            "fight_date": context["fight_date"],
            "weight_class": context["weight_class"],
            "title_fight": bool(title_fight),
        },
        "fighters": {
            "red": context["red_fighter"],
            "blue": context["blue_fighter"],
        },
        "head_to_head": context["head_to_head"],
        "analysis": {
            "top_contributions": _aggregate_contributions(
                transformed_feature_names,
                contributions[:-1],
            ),
            "bias": float(contributions[-1]),
            "global_feature_importance": _load_feature_importance(model_dir),
            "model_metrics": _load_metrics(model_dir),
            "explanation_note": (
                "top_contributions moves the raw model log-odds toward red or blue. "
                "Historical fight statistics are calculated only from fights before fight_date."
            ),
        },
    }
