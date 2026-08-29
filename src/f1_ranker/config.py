from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = PROJECT_ROOT / "F1" / "CSVs"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class DatasetSpec:
    """Define el archivo y las columnas requeridas por cada dataset."""

    filename: str
    required_columns: tuple[str, ...]


DATASET_SPECS = {
    "results": DatasetSpec(
        filename="results.csv",
        required_columns=(
            "raceId",
            "driverId",
            "constructorId",
            "grid",
            "rank",
            "positionOrder",
            "statusId",
        ),
    ),
    "races": DatasetSpec(
        filename="races.csv",
        required_columns=("raceId", "year", "round", "circuitId", "date"),
    ),
    "qualifying": DatasetSpec(
        filename="qualifying.csv",
        required_columns=(
            "raceId",
            "driverId",
            "constructorId",
            "position",
            "q1",
            "q2",
            "q3",
        ),
    ),
    "pit_stops": DatasetSpec(
        filename="pit_stops.csv",
        required_columns=("raceId", "driverId", "milliseconds"),
    ),
    "lap_times": DatasetSpec(
        filename="lap_times.csv",
        required_columns=("raceId", "driverId", "lap", "milliseconds"),
    ),
    "driver_standings": DatasetSpec(
        filename="driver_standings.csv",
        required_columns=("raceId", "driverId", "points", "position", "wins"),
    ),
    "constructor_standings": DatasetSpec(
        filename="constructor_standings.csv",
        required_columns=(
            "raceId",
            "constructorId",
            "points",
            "position",
            "wins",
        ),
    ),
    "circuits": DatasetSpec(
        filename="circuits.csv",
        required_columns=("circuitId", "lat", "lng", "alt"),
    ),
}
