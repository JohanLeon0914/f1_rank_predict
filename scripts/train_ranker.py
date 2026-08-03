import _path_setup  # noqa: F401

from f1_ranker.data_loading import load_training_data
from f1_ranker.feature_engineering import build_model_dataset
from f1_ranker.training import train_ranker


def main():
    datasets = load_training_data()
    dataset = build_model_dataset(datasets)
    result = train_ranker(dataset)

    print("Entrenamiento completado.")
    print("Splits:")
    for key, value in result["splits"].items():
        print(f"  {key}: {value}")

    print("\nMetricas:")
    for split_name, metrics in result["metrics"].items():
        print(f"\n{split_name}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    print("\nTop 15 feature importances:")
    print(
        result["feature_importance"]
        .head(15)
        .to_string(index=False, formatters={"importance_pct": "{:.2%}".format})
    )

    print("\nArtefactos guardados en models/:")
    print("  xgb_ranker.json")
    print("  feature_artifacts.joblib")
    print("  metrics.joblib")
    print("  feature_importance.csv")


if __name__ == "__main__":
    main()
