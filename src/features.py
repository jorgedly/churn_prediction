"""
features.py
Ingeniería de características: derivación de fechas, encoding y escalamiento.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CAT_FEATURES = [
    "product_category",
    "ORDER_TASK_JURISDICTION",
    "FULFILLMENT_LEVEL",
    "TAX_DESIGNATION",
    "FORMATION_STATUS",
    "DEVICE_TYPE",
    "PURCHASE_SOURCE",
    "naics_sector",
]

NUM_FEATURES = [
    "account_age_at_order_days",
    "contract_duration_days",
    "has_term",
    "order_month",
]

FEATURE_COLS = CAT_FEATURES + NUM_FEATURES

# Columnas que se dropean antes del modelado
DROP_COLS = [
    "ORDER_TASK_UUID",
    "ORDER_TASK_STATUS",
    "ORDER_TASK_TYPE",
    "CANCEL_REASON",
    "REFUND_CANCEL_REASON_DEFINITION_NAME",
    "TERM_START_DATE",
    "TERM_END_DATE",
    "OT_CREATED_DATETIME",
    "ACCOUNT_CREATED_DATETIME",
    "NAICS_CODE",
    "ACCOUNT_TYPE",
    "ACCOUNT_SOURCE",
    "BUSINESS_ENTITY_TYPE",
    "REFERRAL_FLAG",
    "INTERNATIONAL_FLAG",
    "DELAYED",
]


def build_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe el DataFrame limpio (salida de clean_data) y retorna
    un DataFrame con todas las features listas para modelado.
    No incluye la columna IS_CANCELED (target).
    """
    df = df_clean.copy()

    # --- Features de fecha ---
    df["account_age_at_order_days"] = (
        df["OT_CREATED_DATETIME"] - df["ACCOUNT_CREATED_DATETIME"]
    ).dt.total_seconds() / 86400

    df["has_term"] = df["TERM_START_DATE"].notna().astype(int)

    df["contract_duration_days"] = (
        df["TERM_END_DATE"] - df["TERM_START_DATE"]
    ).dt.total_seconds() / 86400
    df["contract_duration_days"] = df["contract_duration_days"].fillna(-1)

    df["order_month"] = df["OT_CREATED_DATETIME"].dt.month

    # --- Sector NAICS (primeros 2 dígitos) ---
    df["naics_sector"] = df["NAICS_CODE"].astype(str).str[:2]
    df["naics_sector"] = df["naics_sector"].replace({"un": "unknown", "na": "unknown"})

    # --- Eliminar columnas no usadas ---
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df[FEATURE_COLS]


def fit_encoder(X_train: pd.DataFrame):
    """Ajusta OrdinalEncoder sobre X_train. Retorna el encoder ajustado."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(X_train[CAT_FEATURES])
    return enc


def apply_encoder(X: pd.DataFrame, enc: OrdinalEncoder) -> pd.DataFrame:
    """Aplica el encoder ya ajustado a un DataFrame."""
    X = X.copy()
    X[CAT_FEATURES] = enc.transform(X[CAT_FEATURES])
    return X


def fit_scaler(X_train_encoded: pd.DataFrame):
    """Ajusta StandardScaler sobre el set de entrenamiento. Retorna el scaler."""
    scaler = StandardScaler()
    scaler.fit(X_train_encoded)
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Aplica el scaler ya ajustado. Retorna DataFrame con las mismas columnas."""
    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )
