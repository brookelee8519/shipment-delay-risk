"""
Trains a late-delivery-risk classifier on pre-shipment order features.

Compares a Logistic Regression baseline against a Random Forest, same
comparison as the Retention Lab project - the point isn't to find the
fanciest model, it's to see whether a more flexible model actually buys
anything over a simple linear baseline.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.prepare_data import load_clean_data, build_features, RAW_PATH


def main():
    df = load_clean_data(RAW_PATH)
    X, y = build_features(df)
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"Base rate (always predict 'late'): {y.mean():.1%} would be right\n")

    # --- Logistic Regression baseline ---
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)
    print(f"Logistic Regression ROC-AUC: {lr_auc:.3f}")

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)
    rf_pred = (rf_proba >= 0.5).astype(int)
    print(f"Random Forest ROC-AUC: {rf_auc:.3f}")
    print(f"Random Forest accuracy @0.5: {accuracy_score(y_test, rf_pred):.3f}")
    print()
    print(classification_report(y_test, rf_pred, target_names=["on-time", "late"]))

    # Feature importance - also a leakage sanity check. If anything here
    # looked like it was really encoding the outcome (the way "popularity"
    # did in the Movie Match box-office model), that's a sign to go back
    # and re-check LEAK_COLUMNS.
    importance = pd.Series(rf.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    print("\nTop 15 features by importance:")
    print(importance.head(15))

    # Collapse the one-hot Shipping Mode columns back into a single
    # importance number per mode, more readable than 4 separate rows.
    mode_cols = [c for c in feature_cols if c.startswith("Shipping Mode_")]
    print(f"\nTotal importance from Shipping Mode alone: {importance[mode_cols].sum():.3f}")

    joblib.dump(rf, "src/delay_model.joblib")
    joblib.dump(feature_cols, "src/feature_cols.joblib")
    importance.to_csv("src/feature_importance.csv", header=["importance"])

    metrics = pd.Series({
        "lr_auc": round(lr_auc, 4),
        "rf_auc": round(rf_auc, 4),
        "rf_accuracy": round(accuracy_score(y_test, rf_pred), 4),
        "base_rate": round(float(y.mean()), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    })
    metrics.to_csv("src/model_metrics.csv", header=["value"])

    print("\nSaved src/delay_model.joblib, feature_cols.joblib, "
          "feature_importance.csv, model_metrics.csv")


if __name__ == "__main__":
    main()
