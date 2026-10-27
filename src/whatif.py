"""
Re-scores a single order under each of the 4 shipping modes, so the app
can answer "if I ship this a different way, does the late-delivery risk
actually change?" instead of just scoring whatever mode was picked.
"""

import pandas as pd

from src.prepare_data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FEATURE_COLS

SHIPPING_MODES = ["Same Day", "First Class", "Second Class", "Standard Class"]


def score_all_modes(order_row: pd.Series, model, feature_cols: list) -> pd.DataFrame:
    """order_row: one row of the raw/clean feature columns (FEATURE_COLS),
    with whatever Shipping Mode it actually shipped under - that value is
    ignored and overwritten for each candidate mode."""
    candidates = []
    for mode in SHIPPING_MODES:
        row = order_row.copy()
        row["Shipping Mode"] = mode
        candidates.append(row)

    cand_df = pd.DataFrame(candidates)[FEATURE_COLS]
    X = pd.get_dummies(cand_df, columns=CATEGORICAL_FEATURES, drop_first=False)
    X = X.reindex(columns=feature_cols, fill_value=0)

    risk = model.predict_proba(X)[:, 1]
    result = pd.DataFrame({"Shipping Mode": SHIPPING_MODES, "late_risk": risk})
    return result.sort_values("late_risk").reset_index(drop=True)
