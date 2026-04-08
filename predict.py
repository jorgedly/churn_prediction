"""
predict.py
Script de predicción: carga el modelo entrenado y genera predicciones
para nuevos datos en formato CSV.

Uso:
    python predict.py --input data/nuevos_datos.csv --output predicciones.csv

El CSV de entrada debe tener las mismas columnas que el dataset original
(sin IS_CANCELED, ya que eso es lo que se predice).
"""

import argparse
import joblib
import pandas as pd
from pathlib import Path

from src.preprocessing import clean_data
from src.features import build_features, apply_encoder, apply_scaler, FEATURE_COLS


MODEL_PATH  = Path("models/best_model.joblib")
ENCODER_PATH = Path("models/encoder.joblib")
SCALER_PATH  = Path("models/scaler.joblib")


def load_artifacts():
    """Carga modelo, encoder y scaler desde disco."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Ejecuta primero el notebook para entrenar y guardar el modelo."
        )
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler  = joblib.load(SCALER_PATH)
    return model, encoder, scaler


def predict(input_path: str, output_path: str) -> None:
    input_path  = Path(input_path)
    output_path = Path(output_path)

    print(f"Cargando datos desde: {input_path}")
    df_raw = pd.read_csv(input_path)

    print(f"Registros recibidos: {len(df_raw)}")

    # Limpieza y feature engineering
    df_clean = clean_data(df_raw)
    X = build_features(df_clean)

    # Cargar artefactos
    model, encoder, scaler = load_artifacts()

    # Transformar
    X_enc    = apply_encoder(X, encoder)
    X_scaled = apply_scaler(X_enc, scaler)

    # Predecir
    predictions      = model.predict(X_scaled)
    probabilities    = model.predict_proba(X_scaled)[:, 1]

    # Construir output
    output_df = df_raw[["ORDER_TASK_UUID"]].copy() if "ORDER_TASK_UUID" in df_raw.columns else pd.DataFrame()
    output_df["predicted_canceled"] = predictions
    output_df["probability_canceled"] = probabilities.round(4)

    output_df.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en: {output_path}")
    print(f"  Cancelaciones predichas: {predictions.sum()} ({predictions.mean()*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicción de cancelación de order tasks")
    parser.add_argument("--input",  required=True,  help="CSV con nuevos datos")
    parser.add_argument("--output", default="predicciones.csv", help="CSV de salida con predicciones")
    args = parser.parse_args()

    predict(args.input, args.output)
