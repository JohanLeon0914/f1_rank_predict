import _path_setup  # noqa: F401

from ufc_predictor.data_loading import load_ufc_data
from ufc_predictor.feature_engineering import build_model_dataset
from ufc_predictor.training import train_ufc_model


def main():
    datasets = load_ufc_data()
    dataset = build_model_dataset(datasets)
    result = train_ufc_model(dataset)

    print("Entrenamiento UFC completado.")
    print("Splits:")
    for key, value in result["splits"].items():
        print(f"  {key}: {value}")

    print("\nMetricas:")
    for split_name, metrics in result["metrics"].items():
        print(f"\n{split_name}")
        for metric_name, value in metrics.items():
            if metric_name == "confusion_matrix":
                print(f"  {metric_name}: {value}")
            else:
                print(f"  {metric_name}: {value:.4f}")

    print("\nTop 20 feature importances:")
    print(
        result["feature_importance"]
        .head(20)
        .to_string(index=False, formatters={"importance_pct": "{:.2%}".format})
    )

    print("\nArtefactos guardados:")
    print("  models/ufc/ufc_winner_xgb.json")
    print("  models/ufc/ufc_feature_artifacts.joblib")
    print("  models/ufc/ufc_metrics.joblib")
    print("  models/ufc/ufc_feature_importance.csv")
    print("  reports/ufc/ufc_feature_importance.svg")
    print("  reports/ufc/ufc_roc_curve.svg")


if __name__ == "__main__":
    main()

