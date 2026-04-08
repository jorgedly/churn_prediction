"""
preprocessing.py
Limpieza de datos y agrupación de ORDER_TASK_TYPE en categorías semánticas.
"""

import re
import pandas as pd


# Columnas que solo existen cuando ya ocurrió la cancelación → leakage
LEAKAGE_COLS = ["CANCEL_REASON", "REFUND_CANCEL_REASON_DEFINITION_NAME"]

# Columnas de baja varianza (casi todos los valores son iguales en este dataset)
LOW_VARIANCE_COLS = [
    "ACCOUNT_TYPE",
    "ACCOUNT_SOURCE",
    "BUSINESS_ENTITY_TYPE",
    "REFERRAL_FLAG",
    "INTERNATIONAL_FLAG",
    "DELAYED",
]

# Columnas que se reemplazan por features derivadas
DATE_COLS = [
    "TERM_START_DATE",
    "TERM_END_DATE",
    "OT_CREATED_DATETIME",
    "ACCOUNT_CREATED_DATETIME",
]


def categorize_product(task_type: str) -> str:
    """
    Agrupa ORDER_TASK_TYPE en categorías semánticas de producto.
    Colapsa variantes por versión (premium_plan_v17, pro_plan_v36, etc.)
    en una sola etiqueta por tipo de producto.
    """
    if pd.isna(task_type):
        return "other"
    t = str(task_type).lower().strip()

    if re.search(r"formation|incorporat", t):
        return "formation"
    if re.search(r"registered_agent|change_of_registered", t):
        return "registered_agent"
    if re.search(r"annual_report|bof_compliance|beneficial_ownership|ny_publication", t):
        return "compliance"
    if re.search(r"premium_plan|pro_plan|basic_plan|starter_plan|worry_free|truic_|ra_pro_plan", t):
        return "subscription_plan"
    if re.search(r"1800_accountant|accountant|bookkeeping|banking", t):
        return "accounting_finance"
    if re.search(r"insurance", t):
        return "insurance"
    if re.search(r"ein|corporate_docs|operating_agreement|business_docs|banking_resolution", t):
        return "tax_documents"
    if re.search(r"domain|website|email|logo|static", t):
        return "digital_web"
    if t in ("rush", "standard", "expedite"):
        return "fulfillment_noise"

    return "other"


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpieza completa al DataFrame crudo:
    - Convierte columnas de fecha a datetime
    - Rellena nulos en categóricos con 'unknown'
    - Crea variable objetivo IS_CANCELED
    - Crea columna product_category
    - Filtra registros no terminales (pending, processing, etc.)

    Retorna DataFrame limpio. Las columnas de leakage se conservan
    temporalmente para el EDA y se eliminan en build_features().
    """
    df = df_raw.copy()

    # Filtrar solo estados terminales (evita mezclar registros en curso)
    terminal_statuses = {"complete", "active", "canceled", "declined"}
    df = df[df["ORDER_TASK_STATUS"].str.lower().isin(terminal_statuses)].copy()

    # Variable objetivo
    df["IS_CANCELED"] = (df["ORDER_TASK_STATUS"].str.lower() == "canceled").astype(int)

    # Parsear fechas
    for col in ["TERM_START_DATE", "TERM_END_DATE", "OT_CREATED_DATETIME", "ACCOUNT_CREATED_DATETIME"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Rellenar nulos en categóricos
    cat_cols = [
        "ORDER_TASK_JURISDICTION", "FULFILLMENT_LEVEL", "TAX_DESIGNATION",
        "FORMATION_STATUS", "DEVICE_TYPE", "PURCHASE_SOURCE",
        "NAICS_CODE", "BUSINESS_ENTITY_TYPE",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # Booleanos → entero
    for col in ["REFERRAL_FLAG", "INTERNATIONAL_FLAG", "DELAYED"]:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, "true": 1, "false": 0}).fillna(0).astype(int)

    # Categoría de producto
    df["product_category"] = df["ORDER_TASK_TYPE"].apply(categorize_product)

    return df
