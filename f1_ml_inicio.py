from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import sklearn
except ModuleNotFoundError as error:
    libreria = error.name
    print(f"Falta instalar la libreria requerida: {libreria}")
    print("Instala las dependencias con:")
    print("  python3 -m pip install -r requirements.txt")
    raise SystemExit(1) from error


CSV_DIR = Path("F1") / "CSVs"


def cargar_csvs(csv_dir=CSV_DIR):
    """Carga todos los CSV disponibles sin crear columnas nuevas."""
    archivos_csv = sorted(csv_dir.glob("*.csv"))

    if not archivos_csv:
        raise FileNotFoundError(f"No se encontraron CSV en: {csv_dir}")

    dataframes = {}
    for archivo in archivos_csv:
        nombre = archivo.stem
        dataframes[nombre] = pd.read_csv(archivo, na_values=["\\N"])

    return dataframes


def mostrar_resumen_dataframe(nombre, dataframe, filas=5):
    print("=" * 80)
    print(f"CSV: {nombre}")
    print(f"Filas: {dataframe.shape[0]} | Columnas: {dataframe.shape[1]}")
    print(f"Columnas: {list(dataframe.columns)}")

    print("\nPrimeras filas:")
    print(dataframe.head(filas))

    print("\nTipos de datos:")
    print(dataframe.dtypes)

    columnas_numericas = dataframe.select_dtypes(include=[np.number]).columns
    if len(columnas_numericas) == 0:
        print("\nNo hay columnas numericas para estadisticas.")
        return

    print("\nEstadisticas con pandas:")
    print(dataframe[columnas_numericas].describe())

    print("\nEstadisticas sencillas con numpy:")
    print(f"Columnas numericas: {list(columnas_numericas)}")
    print(dataframe[columnas_numericas].agg([np.nanmean, np.nanmin, np.nanmax]))


def main():
    print("Proyecto inicial de ML para F1")
    print(f"Version de scikit-learn disponible: {sklearn.__version__}")
    print(f"Leyendo CSV desde: {CSV_DIR}")

    dataframes = cargar_csvs()
    print(f"\nCSVs importados: {list(dataframes.keys())}\n")

    for nombre, dataframe in dataframes.items():
        mostrar_resumen_dataframe(nombre, dataframe)


if __name__ == "__main__":
    main()
