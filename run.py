"""
run.py
Script de setup y ejecución completa del proyecto.
Instala dependencias, prepara datos y ejecuta el notebook.

Uso:
    python run.py                         # usa data/ot.csv existente
    python run.py --generar               # genera 5000 filas de datos sintéticos
    python run.py --generar --n 10000     # genera N filas de datos sintéticos
    python run.py --notebook presentacion # ejecuta el notebook corto (presentacion.ipynb)
    python run.py --mlflow                # abre la UI de MLflow al finalizar
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).parent


def run(cmd: list[str], desc: str = "") -> int:
    if desc:
        print(f"\n{'─'*50}")
        print(f"  {desc}")
        print(f"{'─'*50}")
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def check_python():
    version = sys.version_info
    if version < (3, 9):
        print(f"⚠️  Python {version.major}.{version.minor} detectado. Se recomienda Python 3.9+")
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")


def install_requirements():
    req = ROOT / "requirements.txt"
    if not req.exists():
        print("⚠️  No se encontró requirements.txt")
        return
    rc = run([sys.executable, "-m", "pip", "install", "-r", str(req), "-q"],
             "Instalando dependencias...")
    if rc == 0:
        print("✅ Dependencias instaladas")
    else:
        print("❌ Error instalando dependencias. Revisa requirements.txt")
        sys.exit(1)


def generate_data(n: int, seed: int):
    data_path = ROOT / "data" / "ot.csv"
    run(
        [sys.executable, "generate_data.py", "--n", str(n), "--seed", str(seed),
         "--output", str(data_path)],
        f"Generando {n:,} filas de datos sintéticos..."
    )
    print(f"✅ Datos guardados en {data_path}")


def check_data():
    data_path = ROOT / "data" / "ot.csv"
    if not data_path.exists():
        print("\n❌ No se encontró data/ot.csv")
        print("   Opciones:")
        print("   1. Coloca tu CSV en data/ot.csv")
        print("   2. Usa datos sintéticos: python run.py --generar")
        sys.exit(1)
    size = data_path.stat().st_size / 1024
    print(f"✅ Dataset encontrado ({size:.0f} KB)")


def execute_notebook(name: str):
    nb_input  = ROOT / f"{name}.ipynb"
    nb_output = ROOT / f"{name}_ejecutado.ipynb"

    if not nb_input.exists():
        print(f"❌ No se encontró {nb_input}")
        sys.exit(1)

    # Verificar que nbconvert esté disponible
    rc = subprocess.run([sys.executable, "-m", "nbconvert", "--version"],
                        capture_output=True).returncode
    if rc != 0:
        run([sys.executable, "-m", "pip", "install", "nbconvert", "-q"], "Instalando nbconvert...")

    rc = run(
        [sys.executable, "-m", "nbconvert", "--to", "notebook",
         "--execute", str(nb_input), "--output", str(nb_output),
         "--ExecutePreprocessor.timeout=600"],
        f"Ejecutando notebook: {name}.ipynb ..."
    )

    if rc == 0:
        print(f"✅ Notebook ejecutado → {nb_output.name}")
    else:
        print(f"❌ Error ejecutando el notebook. Revisa las celdas manualmente.")
        sys.exit(1)


def open_mlflow():
    print("\n" + "─"*50)
    print("  Iniciando MLflow UI")
    print("  Abre en tu navegador: http://localhost:5000")
    print("  (Ctrl+C para detener)")
    print("─"*50)
    subprocess.run(["mlflow", "ui"], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Setup y ejecución completa del proyecto churn prediction"
    )
    parser.add_argument("--generar", action="store_true",
                        help="Genera datos sintéticos en lugar de usar datos reales")
    parser.add_argument("--n",       type=int, default=5000,
                        help="Número de filas a generar con --generar (default: 5000)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Semilla aleatoria para --generar (default: 42)")
    parser.add_argument("--notebook", type=str, default="order_task_churn_prediction",
                        choices=["order_task_churn_prediction", "presentacion"],
                        help="Notebook a ejecutar (default: completo)")
    parser.add_argument("--mlflow",  action="store_true",
                        help="Abre la UI de MLflow al finalizar")
    parser.add_argument("--skip-install", action="store_true",
                        help="Omite la instalación de dependencias")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  Churn Prediction — Predicción de Cancelación de Servicios")
    print("="*50)

    check_python()

    if not args.skip_install:
        install_requirements()

    if args.generar:
        generate_data(args.n, args.seed)
    else:
        check_data()

    execute_notebook(args.notebook)

    if args.mlflow:
        open_mlflow()
    else:
        print("\n" + "="*50)
        print("  ✅ Proyecto ejecutado correctamente.")
        print("  Para ver experimentos: python run.py --mlflow")
        print("  Para predicciones:     python predict.py --input data/ot.csv --output out.csv")
        print("="*50 + "\n")


if __name__ == "__main__":
    main()
