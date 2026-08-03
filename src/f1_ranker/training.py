"""Entrenamiento del modelo XGBoost Ranker."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRanker

from f1_ranker.config import MODEL_DIR
from f1_ranker.evaluation import evaluate_rankings
from f1_ranker.feature_engineering import (
    DATE_COLUMN,
    GROUP_COLUMN,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
)


def _clean_transformed_feature_name(name: str) -> str:
    """Normaliza nombres emitidos por ColumnTransformer."""
    if name.startswith("numeric__"):
        return name.replace("numeric__", "", 1)
    if name.startswith("categorical__"):
        raw_name = name.replace("categorical__", "", 1)
        if raw_name.startswith("circuitId_"):
            return "circuitId"
        return raw_name
    return name


def build_feature_importance_report(model: XGBRanker, preprocessor) -> pd.DataFrame:
    """Devuelve importancia agregada por feature original."""
    transformed_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    report = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "original_feature": [
                _clean_transformed_feature_name(name) for name in transformed_names
            ],
            "importance": importances,
        }
    )
    grouped = (
        report.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    grouped["importance_pct"] = grouped["importance"] / grouped["importance"].sum()
    return grouped


def temporal_split(
    dataset: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide por carreras en orden temporal, sin mezclar futuro en train."""
    races = (
        dataset[["raceId", DATE_COLUMN]]
        .drop_duplicates()
        .sort_values([DATE_COLUMN, "raceId"])
        .reset_index(drop=True)
    )
    n_races = len(races)
    train_end = max(1, int(n_races * train_ratio))
    validation_end = max(train_end + 1, int(n_races * (train_ratio + validation_ratio)))

    train_races = set(races.iloc[:train_end]["raceId"])
    validation_races = set(races.iloc[train_end:validation_end]["raceId"])
    test_races = set(races.iloc[validation_end:]["raceId"])

    train = dataset[dataset["raceId"].isin(train_races)].copy()
    validation = dataset[dataset["raceId"].isin(validation_races)].copy()
    test = dataset[dataset["raceId"].isin(test_races)].copy()

    return train, validation, test


def sort_for_ranker(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ordena filas para que los grupos de XGBoost queden contiguos."""
    return dataframe.sort_values([GROUP_COLUMN, TARGET_COLUMN]).reset_index(drop=True)


def make_group_sizes(dataframe: pd.DataFrame) -> np.ndarray:
    """Genera el parametro group: cantidad de pilotos por raceId."""
    return dataframe.groupby(GROUP_COLUMN, sort=False).size().to_numpy()


def make_relevance_labels(dataframe: pd.DataFrame) -> np.ndarray:
    """Convierte positionOrder a relevancia, porque XGBoost maximiza el label."""
    max_position_by_race = dataframe.groupby(GROUP_COLUMN)[TARGET_COLUMN].transform("max")
    return (max_position_by_race + 1 - dataframe[TARGET_COLUMN]).to_numpy()


def build_ranker() -> XGBRanker:
    """Crea el modelo base de Learning to Rank."""
    return XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg",
        ndcg_exp_gain=False,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        tree_method="hist",
    )


def train_ranker(
    dataset: pd.DataFrame,
    model_dir: Path = MODEL_DIR,
) -> dict[str, object]:
    """Entrena, evalua y guarda modelo + transformaciones reutilizables."""
    train, validation, test = temporal_split(dataset)
    train = sort_for_ranker(train)
    validation = sort_for_ranker(validation)
    test = sort_for_ranker(test)

    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(train[MODEL_FEATURE_COLUMNS])
    x_validation = preprocessor.transform(validation[MODEL_FEATURE_COLUMNS])
    x_test = preprocessor.transform(test[MODEL_FEATURE_COLUMNS])

    y_train = make_relevance_labels(train)
    y_validation = make_relevance_labels(validation)
    train_group = make_group_sizes(train)
    validation_group = make_group_sizes(validation)

    model = build_ranker()
    model.fit(
        x_train,
        y_train,
        group=train_group,
        eval_set=[(x_validation, y_validation)],
        eval_group=[validation_group],
        verbose=False,
    )

    validation_scores = model.predict(x_validation)
    test_scores = model.predict(x_test)

    metrics = {
        "validation": evaluate_rankings(
            validation["raceId"], validation[TARGET_COLUMN], validation_scores
        ),
        "test": evaluate_rankings(test["raceId"], test[TARGET_COLUMN], test_scores),
    }
    feature_importance = build_feature_importance_report(model, preprocessor)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / "xgb_ranker.json")
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "categorical_note": "circuitId se codifica como categorica con OneHotEncoder.",
        },
        model_dir / "feature_artifacts.joblib",
    )
    joblib.dump(metrics, model_dir / "metrics.joblib")
    feature_importance.to_csv(model_dir / "feature_importance.csv", index=False)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "splits": {
            "train_races": train["raceId"].nunique(),
            "validation_races": validation["raceId"].nunique(),
            "test_races": test["raceId"].nunique(),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
    }


def load_trained_ranker(model_dir: Path = MODEL_DIR) -> tuple[XGBRanker, dict[str, object]]:
    """Carga modelo y artefactos guardados."""
    model = build_ranker()
    model.load_model(model_dir / "xgb_ranker.json")
    artifacts = joblib.load(model_dir / "feature_artifacts.joblib")
    return model, artifacts
