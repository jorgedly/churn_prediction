"""
generate_data.py
Genera un dataset sintético en el formato exacto del CSV de producción.
Útil para probar el pipeline sin datos reales.

Uso:
    python generate_data.py --n 5000 --output data/ot.csv
    python generate_data.py --n 1000 --output data/muestra.csv --seed 99
"""

import argparse
import uuid
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta


# --- Catálogos de valores posibles (basados en producción) ---

ORDER_TASK_TYPES = [
    "llc_formation", "corp_formation", "nonprofit_formation",
    "registered_agent_service", "change_of_registered_agent",
    "llc_annual_report", "annual_report", "ongoing_bof_compliance",
    "beneficial_ownership_filing", "ny_publication",
    "premium_plan_v27", "premium_plan_v36", "premium_plan_v37",
    "pro_plan_v16", "pro_plan_v36", "pro_plan_v37",
    "basic_plan_v27", "starter_plan_v27", "worry_free_service",
    "truic_premium_plan_v24",
    "1800_accountant", "bookkeeping", "banking_resolution",
    "insurance",
    "ein_creation", "ein_business_docs_bundle", "corporate_docs",
    "operating_agreement",
    "domain_name_reg", "domain_name_privacy", "static_website",
    "basic_email", "logo_kit",
]

# Pesos aproximados de cancelación por tipo (para que sea realista)
CANCEL_WEIGHTS = {
    "llc_formation": 0.05, "corp_formation": 0.05, "nonprofit_formation": 0.06,
    "registered_agent_service": 0.18, "change_of_registered_agent": 0.10,
    "llc_annual_report": 0.08, "annual_report": 0.08, "ongoing_bof_compliance": 0.25,
    "beneficial_ownership_filing": 0.30, "ny_publication": 0.12,
    "premium_plan_v27": 0.35, "premium_plan_v36": 0.33, "premium_plan_v37": 0.31,
    "pro_plan_v16": 0.38, "pro_plan_v36": 0.36, "pro_plan_v37": 0.34,
    "basic_plan_v27": 0.40, "starter_plan_v27": 0.42, "worry_free_service": 0.28,
    "truic_premium_plan_v24": 0.32,
    "1800_accountant": 0.20, "bookkeeping": 0.22, "banking_resolution": 0.15,
    "insurance": 0.18,
    "ein_creation": 0.07, "ein_business_docs_bundle": 0.08, "corporate_docs": 0.09,
    "operating_agreement": 0.10,
    "domain_name_reg": 0.45, "domain_name_privacy": 0.43, "static_website": 0.50,
    "basic_email": 0.48, "logo_kit": 0.20,
}

JURISDICTIONS = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
]

FULFILLMENT_LEVELS = ["standard", "rush", "expedite"]
FULFILLMENT_WEIGHTS = [0.45, 0.40, 0.15]

ENTITY_TYPES = ["llc", "corp", "sole_prop", "partnership", "nonprofit"]
ENTITY_WEIGHTS = [0.70, 0.15, 0.07, 0.05, 0.03]

FORMATION_STATUSES = ["formed", "not_formed", "dissolved"]
FORMATION_WEIGHTS = [0.80, 0.15, 0.05]

TAX_DESIGNATIONS = ["sole_prop", "partnership", "c_corp", "s_corp", "unknown"]
TAX_WEIGHTS = [0.55, 0.18, 0.08, 0.12, 0.07]

DEVICE_TYPES = ["desktop", "mobile", None]
DEVICE_WEIGHTS = [0.52, 0.40, 0.08]

PURCHASE_SOURCES = ["flow", "shop", "cust_dashboard", "app_products_suite", None]
PURCHASE_WEIGHTS = [0.45, 0.30, 0.15, 0.05, 0.05]

CANCEL_REASONS = [
    "charge_failure", "not_in_business", "chose_competitor",
    "incomplete", "downgrade", "product_unsupported",
    "system_downgrade", "changed_agent", "customer_not_ready_yet",
    "product_not_using", "system_error", "took_too_long",
]

NAICS_CODES = [
    "541110", "541611", "722513", "531110", "531390",
    "459999", "458110", "812990", "541990", "237990",
    "238990", "455110", "811111", "813920", "522320",
    None, None, None,  # ~33% nulos como en producción
]


def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def generate_row(rng: random.Random) -> dict:
    now = datetime(2026, 4, 1, tzinfo=timezone.utc)
    five_years_ago = datetime(2019, 1, 1, tzinfo=timezone.utc)

    task_type = rng.choice(ORDER_TASK_TYPES)
    cancel_prob = CANCEL_WEIGHTS.get(task_type, 0.20)

    # Fechas base
    account_created = random_date(five_years_ago, now - timedelta(days=1))
    ot_created = random_date(account_created, now)

    # ¿Tiene término (servicios recurrentes)?
    is_recurring = any(k in task_type for k in ["plan", "service", "agent", "1800", "insurance", "domain", "email", "website"])
    if is_recurring:
        term_start = ot_created + timedelta(days=rng.randint(0, 30))
        term_end   = term_start + timedelta(days=365 * rng.choice([1, 2]))
    else:
        term_start = None
        term_end   = None

    # Status y cancelación
    canceled = rng.random() < cancel_prob
    if canceled:
        status = "canceled"
        cancel_reason = rng.choice(CANCEL_REASONS)
        refund_reason = rng.choice(CANCEL_REASONS + [None])
    else:
        status = rng.choices(["complete", "active"], weights=[0.65, 0.35])[0]
        cancel_reason = None
        refund_reason = None

    jurisdiction = rng.choice(JURISDICTIONS)
    fulfillment  = rng.choices(FULFILLMENT_LEVELS, weights=FULFILLMENT_WEIGHTS)[0]
    entity_type  = rng.choices(ENTITY_TYPES,       weights=ENTITY_WEIGHTS)[0]
    formation    = rng.choices(FORMATION_STATUSES, weights=FORMATION_WEIGHTS)[0]
    tax_desig    = rng.choices(TAX_DESIGNATIONS,   weights=TAX_WEIGHTS)[0]
    device       = rng.choices(DEVICE_TYPES,       weights=DEVICE_WEIGHTS)[0]
    purchase_src = rng.choices(PURCHASE_SOURCES,   weights=PURCHASE_WEIGHTS)[0]
    naics        = rng.choice(NAICS_CODES)

    def fmt(dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S.000 Z") if dt else None

    return {
        "ORDER_TASK_UUID":                    str(uuid.uuid4()),
        "ORDER_TASK_STATUS":                  status,
        "IS_CANCELED":                        1 if canceled else 0,
        "ORDER_TASK_TYPE":                    task_type,
        "ORDER_TASK_JURISDICTION":            jurisdiction,
        "FULFILLMENT_LEVEL":                  fulfillment,
        "DELAYED":                            str(rng.random() < 0.05).lower(),
        "TERM_START_DATE":                    fmt(term_start),
        "TERM_END_DATE":                      fmt(term_end),
        "OT_CREATED_DATETIME":                fmt(ot_created),
        "CANCEL_REASON":                      cancel_reason,
        "REFUND_CANCEL_REASON_DEFINITION_NAME": refund_reason,
        "ACCOUNT_TYPE":                       "standard",
        "ACCOUNT_SOURCE":                     rng.choices(["web", "api"], weights=[0.95, 0.05])[0],
        "REFERRAL_FLAG":                      str(rng.random() < 0.08).lower(),
        "INTERNATIONAL_FLAG":                 str(rng.random() < 0.02).lower(),
        "ACCOUNT_CREATED_DATETIME":           fmt(account_created),
        "BUSINESS_ENTITY_TYPE":               entity_type,
        "FORMATION_STATUS":                   formation,
        "TAX_DESIGNATION":                    tax_desig,
        "NAICS_CODE":                         naics,
        "DEVICE_TYPE":                        device,
        "PURCHASE_SOURCE":                    purchase_src,
    }


def generate_dataset(n: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = [generate_row(rng) for _ in range(n)]
    df = pd.DataFrame(rows)
    cancel_rate = df["IS_CANCELED"].mean()
    print(f"Dataset generado: {n:,} filas | Tasa de cancelación: {cancel_rate:.1%}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera datos sintéticos para churn prediction")
    parser.add_argument("--n",      type=int,   default=5000,          help="Número de filas (default: 5000)")
    parser.add_argument("--output", type=str,   default="data/ot.csv", help="Ruta del CSV de salida")
    parser.add_argument("--seed",   type=int,   default=42,            help="Semilla aleatoria para reproducibilidad")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(args.n, args.seed)
    df.to_csv(args.output, index=False)
    print(f"Guardado en: {args.output}")
