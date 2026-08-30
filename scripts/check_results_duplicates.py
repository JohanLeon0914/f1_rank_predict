import _path_setup  # noqa: F401

from f1_ranker.data_loading import (
    DataValidationError,
    build_duplicate_participant_report,
    find_duplicate_participants,
    load_raw_data,
    load_training_data,
    remove_exact_duplicate_rows,
)


def main():
    datasets = load_raw_data()
    results = datasets["results"]

    duplicate_rows = find_duplicate_participants(results)
    duplicate_report = build_duplicate_participant_report(results)
    cleaned_results = remove_exact_duplicate_rows(results)

    exact_duplicate_rows = len(results) - len(cleaned_results)

    print("Reporte de duplicados en results.csv original")
    print(f"Filas duplicadas por raceId + driverId: {len(duplicate_rows)}")
    print(f"Pares duplicados raceId + driverId: {len(duplicate_report)}")
    print(f"Filas completamente identicas eliminables: {exact_duplicate_rows}")

    if duplicate_report.empty:
        print("\nNo hay duplicados por raceId + driverId.")
    else:
        print("\nColumnas que difieren por duplicado:")
        print(duplicate_report.to_string(index=False))

        print("\nEjemplos completos de 3 duplicados:")
        example_keys = duplicate_report[["raceId", "driverId"]].head(3)
        for row in example_keys.itertuples(index=False):
            example = duplicate_rows[
                (duplicate_rows["raceId"] == row.raceId)
                & (duplicate_rows["driverId"] == row.driverId)
            ]
            print(f"\nraceId={row.raceId}, driverId={row.driverId}")
            print(example.to_string(index=False))

    print("\nChequeo de bloqueo para entrenamiento:")
    try:
        load_training_data()
        print("OK: existe exactamente una fila por raceId + driverId.")
    except DataValidationError as error:
        print(str(error))


if __name__ == "__main__":
    main()
