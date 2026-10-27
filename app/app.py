"""
Streamlit dashboard for the shipment delay-risk project.

Run with:
    streamlit run app/app.py

Expects the pipeline to have already been run once:
    python3 -m src.prepare_data
    python3 -m src.train_model
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prepare_data import load_clean_data, build_features, FEATURE_COLS, RAW_PATH
from src.whatif import score_all_modes, SHIPPING_MODES

ROOT = os.path.join(os.path.dirname(__file__), "..")

st.set_page_config(page_title="Shipment Delay Risk", page_icon="📦", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(ROOT, "src/delay_model.joblib"))
    feature_cols = joblib.load(os.path.join(ROOT, "src/feature_cols.joblib"))
    return model, feature_cols


@st.cache_data
def load_scored_data():
    df = load_clean_data(os.path.join(ROOT, RAW_PATH))
    X, _ = build_features(df)
    model, feature_cols = load_artifacts()
    X = X.reindex(columns=feature_cols, fill_value=0)
    df["predicted_risk"] = model.predict_proba(X)[:, 1]
    # Sample down for the app - no need to carry all 170k+ rows into the
    # browser session just to let someone pick one order to look at.
    return df.sample(n=min(len(df), 20000), random_state=7).reset_index(drop=True)


model, feature_cols = load_artifacts()
df = load_scored_data()

st.title("📦 Shipment Delay Risk")
st.caption(
    "Predicts whether a shipment will arrive late, using only what's known "
    "at order time (no peeking at the actual delivery outcome), and shows "
    "how the risk would change under a different shipping mode. Built on "
    "the real DataCo Smart Supply Chain dataset. Each row here is one "
    "order line item (an order with 3 products is 3 rows) - that's the "
    "level the dataset actually tracks shipping outcomes at."
)

tab_lookup, tab_whatif, tab_insights = st.tabs(
    ["🔍 Order Lookup", "🔁 What-If: Shipping Mode", "📊 Model Insights"]
)

# ---------------- Order Lookup ----------------
with tab_lookup:
    col_select, col_random = st.columns([3, 1])
    with col_select:
        order_id = st.selectbox("Choose an order item", df["Order Item Id"].tolist())
    with col_random:
        if st.button("🎲 Random order"):
            order_id = int(np.random.choice(df["Order Item Id"].tolist()))
            st.session_state["picked_order"] = order_id

    if "picked_order" in st.session_state:
        order_id = st.session_state["picked_order"]

    row = df[df["Order Item Id"] == order_id].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted late risk", f"{row['predicted_risk']:.0%}")
    col2.metric("Shipping mode chosen", row["Shipping Mode"])
    col3.metric("Order region", row["Order Region"])
    col4.metric("Actual outcome", "Late" if row["Late_delivery_risk"] == 1 else "On time")

    st.caption(
        "\"Actual outcome\" is shown for reference only - it's never a "
        "model input, since it isn't known until after the order ships."
    )

    with st.expander("Order details"):
        details = row[[
            "Category Name", "Department Name", "Customer Segment",
            "Order Item Quantity", "Order Item Discount Rate",
            "Order Item Product Price", "Market", "Order Country",
            "Product Name",
        ]]
        st.table(details)

# ---------------- What-If ----------------
with tab_whatif:
    st.subheader("How would a different shipping mode change the risk?")
    st.caption(
        "Keeps everything else about the order fixed (region, category, "
        "quantity, discount, order date) and re-scores it under each of "
        "the 4 shipping modes."
    )

    order_id_wi = st.selectbox(
        "Choose an order item", df["Order Item Id"].tolist(), key="whatif_order"
    )
    row_wi = df[df["Order Item Id"] == order_id_wi].iloc[0]
    order_feats = row_wi[FEATURE_COLS]

    results = score_all_modes(order_feats, model, feature_cols)
    results["late_risk_pct"] = (results["late_risk"] * 100).round(1)

    best = results.iloc[0]
    actual_mode = row_wi["Shipping Mode"]
    st.success(
        f"**Lowest predicted risk: {best['Shipping Mode']}** "
        f"({best['late_risk_pct']}% chance of being late). "
        f"This order actually shipped **{actual_mode}**."
    )

    st.bar_chart(results.set_index("Shipping Mode")["late_risk_pct"])
    st.dataframe(
        results.rename(columns={"late_risk_pct": "Predicted late risk (%)"})[
            ["Shipping Mode", "Predicted late risk (%)"]
        ],
        use_container_width=True, hide_index=True,
    )

# ---------------- Model Insights ----------------
with tab_insights:
    st.subheader("What drives late-delivery risk")
    importance = pd.read_csv(
        os.path.join(ROOT, "src/feature_importance.csv"), index_col=0
    ).squeeze()
    top = importance.sort_values(ascending=True).tail(12)
    st.bar_chart(top)
    st.caption(
        "Shipping Mode alone accounts for the large majority of the "
        "model's predictive power - see the README for why."
    )

    st.subheader("Late delivery rate by shipping mode (actual data)")
    mode_rates = (
        df.groupby("Shipping Mode")["Late_delivery_risk"].mean().sort_values() * 100
    )
    st.bar_chart(mode_rates)

    st.subheader("Model performance")
    metrics = pd.read_csv(
        os.path.join(ROOT, "src/model_metrics.csv"), index_col=0
    ).squeeze()
    m1, m2, m3 = st.columns(3)
    m1.metric("Random Forest ROC-AUC", f"{float(metrics['rf_auc']):.3f}")
    m2.metric("Logistic Regression ROC-AUC", f"{float(metrics['lr_auc']):.3f}")
    m3.metric("Base rate (always 'late')", f"{float(metrics['base_rate']):.1%}")
