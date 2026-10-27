"""
Loads the raw DataCo Supply Chain CSV and builds a clean, leakage-free
feature table for predicting late delivery risk.

The dataset logs the *actual* outcome of each shipment (Days for shipping
(real), Delivery Status, shipping date) right next to the order-time
features. A model trained with those columns in the feature set would
just be reading the answer key - it would "predict" lateness using data
that doesn't exist until the package has already arrived. Everything in
LEAK_COLUMNS below is excluded from X for exactly that reason.

Canceled orders are dropped too: "Shipping canceled" isn't really an
on-time delivery, it's a different outcome entirely (the dataset happens
to code it as Late_delivery_risk = 0, which would just teach the model
that cancellations look like success).
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/DataCoSupplyChainDataset.csv"

# Known only AFTER the order ships / is resolved - must never be features.
LEAK_COLUMNS = [
    "Days for shipping (real)",
    "Delivery Status",
    "Late_delivery_risk",  # this is the target, re-added separately
    "shipping date (DateOrders)",
    "Benefit per order",
    "Sales per customer",
    "Order Profit Per Order",
    "Order Item Profit Ratio",
    "Order Status",
]

# Identifying/PII columns - the dataset ships with fake customer records
# (including a plaintext "Customer Password" column). None of this has any
# business being in a model, and it shouldn't be displayed either.
PII_COLUMNS = [
    "Customer Email", "Customer Fname", "Customer Lname", "Customer Password",
    "Customer Street", "Customer Zipcode", "Customer Id", "Order Customer Id",
    "Order Zipcode", "Product Description", "Product Image",
]

CATEGORICAL_FEATURES = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Category Name",
    "Department Name",
    "Customer Segment",
    "Type",
]

NUMERIC_FEATURES = [
    "Order Item Quantity",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "order_weekday",
    "order_month",
]

FEATURE_COLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COL = "Late_delivery_risk"

# Columns kept around for the app (order lookup, display) but not fed to
# the model.
DISPLAY_COLS = [
    "Order Id", "Order Item Id", "Customer City", "Customer Country",
    "Customer State", "Order City", "Order Country", "Order State",
    "Product Name",
]


def load_clean_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")

    # Drop canceled shipments - they never had a real delivery outcome.
    df = df[df["Delivery Status"] != "Shipping canceled"].copy()

    df["order_date_parsed"] = pd.to_datetime(
        df["order date (DateOrders)"], errors="coerce"
    )
    df["order_weekday"] = df["order_date_parsed"].dt.dayofweek
    df["order_month"] = df["order_date_parsed"].dt.month

    df = df.drop(columns=PII_COLUMNS, errors="ignore")

    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=False)
    y = df[TARGET_COL].astype(int)
    return X, y


if __name__ == "__main__":
    df = load_clean_data()
    print(f"Loaded {len(df):,} rows after dropping canceled shipments")
    print(f"Late delivery rate: {df['Late_delivery_risk'].mean():.1%}")
    X, y = build_features(df)
    print(f"Feature matrix: {X.shape}")
    df.to_csv("data/clean_orders.csv", index=False)
    print("Saved data/clean_orders.csv")
