"""
evaluation.py
Funciones de evaluación, métricas y visualizaciones de resultados.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_model(name: str, clf, X_eval, y_eval, use_proba: bool = True) -> dict:
    """
    Evalúa un modelo clasificador y retorna un diccionario con métricas.

    Args:
        name:       Nombre del modelo (para el reporte).
        clf:        Modelo entrenado (sklearn Pipeline o estimator).
        X_eval:     Features de evaluación.
        y_eval:     Target real.
        use_proba:  Si True, usa predict_proba para ROC-AUC.
                    Usar False para Perceptrón (no tiene predict_proba).
    """
    y_pred = clf.predict(X_eval)

    roc_auc = None
    if use_proba and hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_eval)[:, 1]
        roc_auc = roc_auc_score(y_eval, y_prob)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_eval, y_pred, target_names=["No cancelado", "Cancelado"]))

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_eval, y_pred),
        "F1_macro": f1_score(y_eval, y_pred, average="macro"),
        "F1_canceled": f1_score(y_eval, y_pred, pos_label=1, average="binary"),
        "ROC_AUC": roc_auc,
    }


def plot_confusion_matrix(name: str, clf, X_eval, y_eval) -> None:
    """Muestra la matriz de confusión de un modelo."""
    y_pred = clf.predict(X_eval)
    cm = confusion_matrix(y_eval, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No cancelado", "Cancelado"],
    )
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de confusión — {name}")
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results: list[dict]) -> None:
    """
    Genera gráfica de barras comparando Accuracy, F1_macro y ROC_AUC
    entre todos los modelos evaluados.
    """
    results_df = pd.DataFrame(results).set_index("Model")
    metrics = ["Accuracy", "F1_macro", "ROC_AUC"]
    plot_df = results_df[metrics].dropna(axis=1)

    ax = plot_df.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Comparación de modelos")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    print("\nTabla de resultados:")
    print(results_df.round(4))
