import _path_setup  # noqa: F401

from f1_ranker.data_loading import load_training_data
from f1_ranker.feature_engineering import build_model_dataset


def main():
    datasets = load_training_data()
    dataset = build_model_dataset(datasets)
    output_path = "reports/model_dataset_preview.csv"
    dataset.head(100).to_csv(output_path, index=False)

    print("Dataset de modelado construido correctamente.")
    print(f"Filas: {len(dataset)}")
    print(f"Carreras: {dataset['raceId'].nunique()}")
    print(f"Pilotos unicos: {dataset['driverId'].nunique()}")
    print(f"Vista previa guardada en: {output_path}")


if __name__ == "__main__":
    main()
