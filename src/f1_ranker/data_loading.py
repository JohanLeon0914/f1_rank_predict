from pathlib import Path

import pandas as pd

from f1_ranker.config import CSV_DIR, DATASET_SPECS, REPORT_DIR, DatasetSpec


class DataValidationError(ValueError):
    """Error usado cuando los CSV no cumplen el contrato del pipeline."""


def read_csv_dataset(csv_dir: Path, spec: DatasetSpec) -> pd.DataFrame:
    """Lee un CSV reemplazando el marcador '\\N' por valores faltantes."""
    path = csv_dir / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")

    return pd.read_csv(path, na_values=["\\N"])


def validate_required_columns(
    dataset_name: str,
    dataframe: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> None:
    """Valida que un DataFrame contenga todas las columnas requeridas."""
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise DataValidationError(
            f"El dataset '{dataset_name}' no contiene columnas requeridas: {missing}"
        )


PARTICIPANT_KEY = ["raceId", "driverId"]


def find_duplicate_participants(results: pd.DataFrame) -> pd.DataFrame:
    """Devuelve todas las filas duplicadas por piloto y carrera."""
    duplicated = results.duplicated(subset=["raceId", "driverId"], keep=False)
    return results.loc[duplicated].sort_values(PARTICIPANT_KEY)


def get_differing_columns(group: pd.DataFrame) -> list[str]:
    """Identifica columnas que no tienen el mismo valor dentro de un duplicado."""
    differing_columns = []
    for column in group.columns:
        unique_values = group[column].drop_duplicates()
        if len(unique_values) > 1:
            differing_columns.append(column)

    return differing_columns


def build_duplicate_participant_report(results: pd.DataFrame) -> pd.DataFrame:
    """Construye un reporte por cada grupo duplicado raceId + driverId.

    El reporte no modifica datos. Indica cuantas filas hay por duplicado y que
    columnas difieren, para decidir reglas manuales de consolidacion.
    """
    duplicates = find_duplicate_participants(results)
    if duplicates.empty:
        return pd.DataFrame(
            columns=["raceId", "driverId", "duplicate_rows", "differing_columns"]
        )

    rows = []
    for (race_id, driver_id), group in duplicates.groupby(PARTICIPANT_KEY, sort=True):
        rows.append(
            {
                "raceId": race_id,
                "driverId": driver_id,
                "duplicate_rows": len(group),
                "differing_columns": ", ".join(get_differing_columns(group)),
            }
        )

    return pd.DataFrame(rows)


def remove_exact_duplicate_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Elimina solo filas completamente identicas en results.csv."""
    return results.drop_duplicates().reset_index(drop=True)


def validate_unique_participants_or_report(
    results: pd.DataFrame,
    report_dir: Path = REPORT_DIR,
) -> pd.DataFrame:
    """Garantiza una fila por raceId + driverId antes de entrenar.

    Si hay filas exactamente identicas, se eliminan. Si despues de eso quedan
    duplicados con diferencias, se genera un reporte y se detiene el pipeline.
    """
    cleaned_results = remove_exact_duplicate_rows(results)
    duplicate_report = build_duplicate_participant_report(cleaned_results)

    if duplicate_report.empty:
        return cleaned_results

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "results_duplicate_participants_report.csv"
    rows_path = report_dir / "results_duplicate_participants_rows.csv"

    duplicate_rows = find_duplicate_participants(cleaned_results)
    duplicate_report.to_csv(report_path, index=False)
    duplicate_rows.to_csv(rows_path, index=False)

    raise DataValidationError(
        "results.csv contiene duplicados no identicos por raceId + driverId. "
        "El entrenamiento queda bloqueado hasta revision manual. "
        f"Reporte: {report_path}. Filas completas: {rows_path}"
    )


def clean_results_for_training(
    results: pd.DataFrame,
    report_dir: Path = REPORT_DIR,
) -> pd.DataFrame:
    """Limpia results.csv para entrenamiento excluyendo duplicados conflictivos.

    Regla acordada:
    - se eliminan filas completamente identicas;
    - si un par raceId + driverId queda duplicado con valores distintos, se
      excluyen todas las filas de ese par y se guarda un reporte.
    """
    cleaned_results = remove_exact_duplicate_rows(results)
    duplicate_report = build_duplicate_participant_report(cleaned_results)

    if duplicate_report.empty:
        return cleaned_results

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "results_duplicate_participants_report.csv"
    rows_path = report_dir / "results_duplicate_participants_rows.csv"

    duplicate_rows = find_duplicate_participants(cleaned_results)
    duplicate_report.to_csv(report_path, index=False)
    duplicate_rows.to_csv(rows_path, index=False)

    duplicate_keys = duplicate_report[PARTICIPANT_KEY]
    training_results = cleaned_results.merge(
        duplicate_keys.assign(_duplicated_pair=True),
        on=PARTICIPANT_KEY,
        how="left",
    )
    training_results = training_results[
        training_results["_duplicated_pair"].isna()
    ].drop(columns=["_duplicated_pair"])

    remaining_duplicates = find_duplicate_participants(training_results)
    if not remaining_duplicates.empty:
        raise DataValidationError(
            "La limpieza no produjo un results.csv unico por raceId + driverId."
        )

    return training_results.reset_index(drop=True)


def load_raw_data(csv_dir: Path = CSV_DIR) -> dict[str, pd.DataFrame]:
    """Carga todos los CSV requeridos por el pipeline.

    Esta funcion no hace feature engineering ni crea columnas nuevas. Solo lee
    los archivos y valida que exista el contrato minimo para etapas posteriores.
    """
    datasets = {}
    for dataset_name, spec in DATASET_SPECS.items():
        dataframe = read_csv_dataset(csv_dir, spec)
        validate_required_columns(dataset_name, dataframe, spec.required_columns)
        datasets[dataset_name] = dataframe

    return datasets


def load_training_data(csv_dir: Path = CSV_DIR) -> dict[str, pd.DataFrame]:
    """Carga datos y aplica la limpieza acordada para entrenamiento."""
    datasets = load_raw_data(csv_dir)
    datasets["results"] = clean_results_for_training(datasets["results"])
    return datasets


def summarize_raw_data(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Devuelve un resumen compacto de filas, columnas y valores faltantes."""
    rows = []
    for name, dataframe in datasets.items():
        rows.append(
            {
                "dataset": name,
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
                "missing_values": int(dataframe.isna().sum().sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def build_data_quality_report(datasets: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Construye un reporte de calidad sin modificar los datos cargados."""
    duplicate_participants = find_duplicate_participants(datasets["results"])
    duplicate_report = build_duplicate_participant_report(datasets["results"])
    exact_duplicate_rows = len(datasets["results"]) - len(remove_exact_duplicate_rows(datasets["results"]))

    return {
        "results_duplicate_participants_count": len(duplicate_participants),
        "results_duplicate_pairs_count": len(duplicate_report),
        "results_exact_duplicate_rows_count": exact_duplicate_rows,
        "results_duplicate_participants_sample": duplicate_participants.head(10),
        "results_duplicate_report": duplicate_report,
    }
