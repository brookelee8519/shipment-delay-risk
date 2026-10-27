"""
Renders the core charts to a static PNG, so the README has something to
show without requiring anyone to install and run Streamlit first.
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.prepare_data import load_clean_data, RAW_PATH

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    df = load_clean_data(RAW_PATH)
    importance = pd.read_csv(
        os.path.join(ROOT, "src/feature_importance.csv"), index_col=0
    ).squeeze().sort_values(ascending=True).tail(10)
    metrics = pd.read_csv(
        os.path.join(ROOT, "src/model_metrics.csv"), index_col=0
    ).squeeze()
    mode_rates = (
        df.groupby("Shipping Mode")["Late_delivery_risk"].mean().sort_values() * 100
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].barh(mode_rates.index, mode_rates.values, color="#4C72B0")
    axes[0].set_xlabel("Late delivery rate (%)")
    axes[0].set_title("Late rate by shipping mode")
    for i, v in enumerate(mode_rates.values):
        axes[0].text(v + 1, i, f"{v:.0f}%", va="center")

    axes[1].barh(importance.index, importance.values, color="#55A868")
    axes[1].set_xlabel("Feature importance")
    axes[1].set_title("Top 10 features (Random Forest)")

    fig.suptitle(
        f"Shipment Delay Risk  |  RF ROC-AUC {float(metrics['rf_auc']):.3f}  |  "
        f"LR ROC-AUC {float(metrics['lr_auc']):.3f}  |  "
        f"Base rate {float(metrics['base_rate']):.1%}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(ROOT, "screenshot.png")
    fig.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
