from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from ufc_predictor.config import UFC_MODEL_DIR, UFC_REPORT_DIR
from ufc_predictor.feature_engineering import (
    DATE_COLUMN,
    TARGET_COLUMN,
    build_preprocessor,
    feature_columns,
)


def temporal_split(
    dataset: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = dataset.sort_values([DATE_COLUMN, "fight_id"]).reset_index(drop=True)
    n_rows = len(ordered)
    train_end = max(1, int(n_rows * train_ratio))
    validation_end = max(train_end + 1, int(n_rows * (train_ratio + validation_ratio)))
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )


def build_classifier() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=450,
        learning_rate=0.035,
        max_depth=3,
        min_child_weight=4,
        subsample=0.88,
        colsample_bytree=0.88,
        reg_lambda=2.0,
        random_state=42,
        tree_method="hist",
    )


def evaluate_binary(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": {
            "true_blue_pred_blue": int(tn),
            "true_blue_pred_red": int(fp),
            "true_red_pred_blue": int(fn),
            "true_red_pred_red": int(tp),
        },
    }


def _clean_feature_name(name: str) -> str:
    if name.startswith("numeric__"):
        return name.replace("numeric__", "", 1)
    if name.startswith("categorical__"):
        return name.replace("categorical__", "", 1)
    return name


def build_feature_importance_report(model: XGBClassifier, preprocessor) -> pd.DataFrame:
    names = preprocessor.get_feature_names_out()
    raw = pd.DataFrame(
        {
            "transformed_feature": names,
            "original_feature": [_clean_feature_name(name) for name in names],
            "importance": model.feature_importances_,
        }
    )
    grouped = (
        raw.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    total = grouped["importance"].sum()
    grouped["importance_pct"] = grouped["importance"] / total if total else 0.0
    return grouped


def _svg_bar_chart(rows: pd.DataFrame, title: str, path: Path, value_column: str = "importance_pct") -> None:
    top = rows.head(20).copy()
    width = 1200
    row_height = 30
    left = 360
    right = 80
    top_pad = 58
    height = top_pad + len(top) * row_height + 35
    max_value = float(top[value_column].max() or 1.0)
    labels = []
    for idx, row in enumerate(top.itertuples()):
        y = top_pad + idx * row_height
        value = float(getattr(row, value_column))
        bar_width = int((width - left - right) * value / max_value)
        label = str(row.original_feature).replace("&", "&amp;")
        pct = f"{value * 100:.2f}%" if value_column.endswith("pct") else f"{value:.4f}"
        labels.append(
            f'<text x="20" y="{y + 19}" font-size="14" fill="#172033">{label}</text>'
            f'<rect x="{left}" y="{y}" width="{bar_width}" height="20" fill="#d71920" />'
            f'<text x="{left + bar_width + 10}" y="{y + 16}" font-size="13" fill="#172033">{pct}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f7f7f2"/>'
        f'<text x="20" y="34" font-size="24" font-family="Arial" font-weight="700" fill="#111827">{title}</text>'
        f'<g font-family="Arial">{"".join(labels)}</g></svg>'
    )
    path.write_text(svg, encoding="utf-8")


def _svg_roc_curve(y_true: pd.Series, probabilities: np.ndarray, auc: float, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    width = 760
    height = 560
    left = 70
    bottom = 500
    plot = 420
    points = [
        f"{left + x * plot:.1f},{bottom - y * plot:.1f}"
        for x, y in zip(fpr, tpr)
    ]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f2"/>
<text x="40" y="36" font-family="Arial" font-size="24" font-weight="700" fill="#111827">UFC ROC Curve - test AUC {auc:.3f}</text>
<line x1="{left}" y1="{bottom}" x2="{left + plot}" y2="{bottom}" stroke="#111827"/>
<line x1="{left}" y1="{bottom}" x2="{left}" y2="{bottom - plot}" stroke="#111827"/>
<line x1="{left}" y1="{bottom}" x2="{left + plot}" y2="{bottom - plot}" stroke="#9ca3af" stroke-dasharray="6 6"/>
<polyline points="{' '.join(points)}" fill="none" stroke="#d71920" stroke-width="3"/>
<text x="{left + plot / 2 - 70}" y="{bottom + 38}" font-family="Arial" font-size="15" fill="#111827">False positive rate</text>
<text x="18" y="{bottom - plot / 2}" transform="rotate(-90 18 {bottom - plot / 2})" font-family="Arial" font-size="15" fill="#111827">True positive rate</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def train_ufc_model(
    dataset: pd.DataFrame,
    model_dir: Path = UFC_MODEL_DIR,
    report_dir: Path = UFC_REPORT_DIR,
) -> dict[str, Any]:
    train, validation, test = temporal_split(dataset)
    columns = feature_columns(dataset)
    preprocessor = build_preprocessor(columns)
    x_train = preprocessor.fit_transform(train[columns])
    x_validation = preprocessor.transform(validation[columns])
    x_test = preprocessor.transform(test[columns])

    model = build_classifier()
    model.fit(
        x_train,
        train[TARGET_COLUMN],
        eval_set=[(x_validation, validation[TARGET_COLUMN])],
        verbose=False,
    )

    validation_proba = model.predict_proba(x_validation)[:, 1]
    test_proba = model.predict_proba(x_test)[:, 1]
    metrics = {
        "validation": evaluate_binary(validation[TARGET_COLUMN], validation_proba),
        "test": evaluate_binary(test[TARGET_COLUMN], test_proba),
    }
    importance = build_feature_importance_report(model, preprocessor)
    splits = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_start": str(train[DATE_COLUMN].min().date()),
        "train_end": str(train[DATE_COLUMN].max().date()),
        "validation_start": str(validation[DATE_COLUMN].min().date()),
        "validation_end": str(validation[DATE_COLUMN].max().date()),
        "test_start": str(test[DATE_COLUMN].min().date()),
        "test_end": str(test[DATE_COLUMN].max().date()),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / "ufc_winner_xgb.json")
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "feature_columns": columns,
            "trained_through": str(dataset[DATE_COLUMN].max().date()),
            "target": TARGET_COLUMN,
            "model_note": "Probabilidad estimada de victoria del peleador rojo en el emparejamiento dado.",
        },
        model_dir / "ufc_feature_artifacts.joblib",
    )
    joblib.dump({"metrics": metrics, "splits": splits}, model_dir / "ufc_metrics.joblib")
    importance.to_csv(model_dir / "ufc_feature_importance.csv", index=False)
    dataset.head(200).to_csv(report_dir / "ufc_model_dataset_preview.csv", index=False)
    pd.DataFrame(metrics).to_json(report_dir / "ufc_metrics.json", indent=2)
    _svg_bar_chart(importance, "UFC Winner Model - Feature Importance", report_dir / "ufc_feature_importance.svg")
    _svg_roc_curve(
        test[TARGET_COLUMN],
        test_proba,
        metrics["test"]["roc_auc"],
        report_dir / "ufc_roc_curve.svg",
    )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "feature_importance": importance,
        "splits": splits,
    }


def load_trained_ufc_model(model_dir: Path = UFC_MODEL_DIR) -> tuple[XGBClassifier, dict[str, Any]]:
    model = build_classifier()
    model.load_model(model_dir / "ufc_winner_xgb.json")
    artifacts = joblib.load(model_dir / "ufc_feature_artifacts.joblib")
    return model, artifacts

