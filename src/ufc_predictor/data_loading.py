from pathlib import Path

import pandas as pd

from ufc_predictor.config import DATASET_SPECS, UFC_CSV_DIR, DatasetSpec


class DataValidationError(ValueError):
    """Error usado cuando los CSV de UFC no cumplen el contrato minimo."""


def read_csv_dataset(csv_dir: Path, spec: DatasetSpec) -> pd.DataFrame:
    path = csv_dir / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")
    return pd.read_csv(path, na_values=["", "NA", "N/A", "\\N"])


def validate_required_columns(
    dataset_name: str,
    dataframe: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise DataValidationError(
            f"El dataset UFC '{dataset_name}' no contiene columnas requeridas: {missing}"
        )


def load_ufc_data(csv_dir: Path = UFC_CSV_DIR) -> dict[str, pd.DataFrame]:
    datasets = {}
    for dataset_name, spec in DATASET_SPECS.items():
        dataframe = read_csv_dataset(csv_dir, spec)
        validate_required_columns(dataset_name, dataframe, spec.required_columns)
        datasets[dataset_name] = dataframe
    return datasets

