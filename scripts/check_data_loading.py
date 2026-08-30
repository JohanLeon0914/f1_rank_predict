import _path_setup  # noqa: F401

from f1_ranker.data_loading import (
    build_data_quality_report,
    load_raw_data,
    summarize_raw_data,
)


def main():
    datasets = load_raw_data()
    summary = summarize_raw_data(datasets)

    print("Carga de datos completada correctamente.")
    print(summary.to_string(index=False))

    results_columns = [
        "raceId",
        "driverId",
        "constructorId",
        "grid",
        "rank",
        "positionOrder",
    ]
    print("\nVista inicial de results.csv:")
    print(datasets["results"][results_columns].head().to_string(index=False))

    quality_report = build_data_quality_report(datasets)
    duplicate_count = quality_report["results_duplicate_participants_count"]
    print("\nReporte de calidad:")
    print(f"Duplicados raceId + driverId en results.csv: {duplicate_count}")
    if duplicate_count:
        print("Muestra de duplicados detectados:")
        print(
            quality_report["results_duplicate_participants_sample"].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
