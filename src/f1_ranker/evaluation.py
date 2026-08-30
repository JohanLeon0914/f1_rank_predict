"""Metricas de evaluacion para rankings de Formula 1."""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score


def _predicted_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    positions = np.empty_like(order)
    positions[order] = np.arange(1, len(scores) + 1)
    return positions


def evaluate_rankings(
    race_ids: pd.Series,
    true_positions: pd.Series,
    scores: np.ndarray,
) -> dict[str, float]:
    """Evalua rankings carrera por carrera y devuelve metricas promedio."""
    frame = pd.DataFrame(
        {
            "raceId": race_ids.to_numpy(),
            "positionOrder": true_positions.to_numpy(),
            "score": scores,
        }
    )

    ndcg_values = []
    spearman_values = []
    kendall_values = []
    mae_values = []
    top1_values = []
    top3_values = []
    top10_values = []

    for _, group in frame.groupby("raceId"):
        if len(group) < 2:
            continue

        y_true_position = group["positionOrder"].to_numpy()
        y_score = group["score"].to_numpy()
        relevance = y_true_position.max() + 1 - y_true_position
        predicted_position = _predicted_positions(y_score)

        ndcg_values.append(ndcg_score([relevance], [y_score]))
        mae_values.append(np.mean(np.abs(predicted_position - y_true_position)))

        spearman = spearmanr(y_true_position, predicted_position).correlation
        kendall = kendalltau(y_true_position, predicted_position).correlation
        if not np.isnan(spearman):
            spearman_values.append(spearman)
        if not np.isnan(kendall):
            kendall_values.append(kendall)

        actual_top1 = set(group.loc[group["positionOrder"] <= 1].index)
        actual_top3 = set(group.loc[group["positionOrder"] <= 3].index)
        actual_top10 = set(group.loc[group["positionOrder"] <= 10].index)
        predicted_order = group.assign(predicted_position=predicted_position)
        predicted_top1 = set(predicted_order.nsmallest(1, "predicted_position").index)
        predicted_top3 = set(predicted_order.nsmallest(3, "predicted_position").index)
        predicted_top10 = set(predicted_order.nsmallest(10, "predicted_position").index)

        top1_values.append(float(bool(actual_top1 & predicted_top1)))
        top3_values.append(len(actual_top3 & predicted_top3) / max(1, len(actual_top3)))
        top10_values.append(len(actual_top10 & predicted_top10) / max(1, len(actual_top10)))

    return {
        "ndcg": float(np.mean(ndcg_values)),
        "spearman": float(np.mean(spearman_values)),
        "kendall_tau": float(np.mean(kendall_values)),
        "mae_position": float(np.mean(mae_values)),
        "top1_accuracy": float(np.mean(top1_values)),
        "top3_accuracy": float(np.mean(top3_values)),
        "top10_accuracy": float(np.mean(top10_values)),
    }
