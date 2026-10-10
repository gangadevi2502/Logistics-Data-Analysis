"""
Week 1 Logistics Analysis Plan
Scenario: reducing late deliveries in an e-commerce supply chain.

Before running:
1. Put DataCoSupplyChainDataset.csv in the same folder as this script.
2. Install: pandas numpy matplotlib scikit-learn scipy
3. Check the field names against DescriptionDataCoSupplyChain.csv.

This script calculates descriptive KPIs and illustrative models. Optimization
costs/capacities are assumptions and must not be treated as real recommendations.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    silhouette_score,
)
from sklearn.cluster import KMeans

DATA_FILE = Path("DataCoSupplyChainDataset.csv")
RANDOM_STATE = 42


def normalise_columns(frame):
    frame = frame.copy()
    frame.columns = (
        frame.columns.astype(str).str.strip().str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)
        .str.strip("_")
    )
    return frame


def load_and_prepare(path=DATA_FILE):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Could not find {path}. Place the DataCo CSV beside this script."
        )

    raw = normalise_columns(pd.read_csv(path, encoding="latin-1"))
    required = [
        "order_status", "order_date_dateorders", "days_for_shipping_real",
        "days_for_shipment_scheduled", "shipping_mode", "order_region",
        "category_name", "customer_segment", "order_item_quantity",
        "order_item_discount_rate", "order_country", "sales",
        "order_profit_per_order",
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            "The following expected columns were not found after column "
            f"normalisation: {missing}. Check DescriptionDataCoSupplyChain.csv."
        )

    print("Raw dataset shape:", raw.shape)
    print("\nColumn types (first 15):\n", raw.dtypes.head(15))
    print("\nColumns with most missing values:\n",
          raw.isna().sum().sort_values(ascending=False).head(10))

    # Deduplicate first, then calculate cancellation rate before shipping-specific filtering.
    df = raw.drop_duplicates().copy()
    cancellation_rate = (
        df["order_status"].astype("string").str.strip().str.upper()
        .eq("CANCELED").mean()
    )
    # Remove direct personal/sensitive fields when present.
    personal_columns = [
        "customer_email", "customer_password", "customer_fname",
        "customer_lname", "customer_street", "customer_zipcode",
    ]
    df = df.drop(columns=[c for c in personal_columns if c in df.columns])

    df["order_date"] = pd.to_datetime(
        df["order_date_dateorders"], errors="coerce"
    )
    if "shipping_date_dateorders" in df.columns:
        df["shipping_date"] = pd.to_datetime(
            df["shipping_date_dateorders"], errors="coerce"
        )

    key_cols = [
        "order_date", "days_for_shipping_real",
        "days_for_shipment_scheduled",
    ]
    df = df.dropna(subset=key_cols).copy()
    df["order_month"] = df["order_date"].dt.month
    df["order_weekday"] = df["order_date"].dt.weekday
    df["delay_days"] = (
        df["days_for_shipping_real"] - df["days_for_shipment_scheduled"]
    )
    df["is_late"] = (df["delay_days"] > 0).astype(int)

    return df, cancellation_rate


def calculate_kpis(df, cancellation_rate):
    if df.empty:
        raise ValueError("No records remain after filtering required shipping fields.")

    sales_total = pd.to_numeric(df["sales"], errors="coerce").sum()
    profit_total = pd.to_numeric(
        df["order_profit_per_order"], errors="coerce"
    ).sum()
    profit_margin = profit_total / sales_total if sales_total else np.nan

    late_only = df.loc[df["delay_days"] > 0, "delay_days"]
    kpis = {
        "On-time delivery rate": (df["delay_days"] <= 0).mean(),
        "Average shipping difference (days)": df["delay_days"].mean(),
        "Average delay among late orders (days)": (
            late_only.mean() if not late_only.empty else 0.0
        ),
        "Average actual shipping time (days)": df["days_for_shipping_real"].mean(),
        "Order cancellation rate (all deduplicated orders)": cancellation_rate,
        "Profit margin": profit_margin,
    }
    print("\nBASELINE KPIs")
    for name, value in kpis.items():
        if pd.isna(value):
            print(f"{name}: unavailable (sales total is zero or missing)")
        elif "rate" in name.lower() or "margin" in name.lower():
            print(f"{name}: {value:.1%}")
        else:
            print(f"{name}: {value:.2f}")

    kpi_by_mode = df.groupby("shipping_mode", dropna=False).agg(
        orders=("is_late", "size"),
        late_rate=("is_late", "mean"),
        avg_shipping_difference=("delay_days", "mean"),
    )
    print("\nKPIs by shipping mode:\n", kpi_by_mode)
    print("\nLate rate by region:\n",
          df.groupby("order_region", dropna=False)["is_late"]
          .mean().sort_values(ascending=False))
    return kpi_by_mode


def exploratory_analysis(df, kpi_by_mode):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    kpi_by_mode["late_rate"].plot(
        kind="bar", ax=axes[0], title="Late rate by shipping mode"
    )
    df["delay_days"].plot(
        kind="hist", bins=20, ax=axes[1], title="Shipping difference (days)"
    )
    df.groupby(df["order_date"].dt.to_period("M")).size().plot(
        ax=axes[2], title="Orders per month"
    )
    axes[0].set_ylabel("Late-order proportion")
    axes[1].set_xlabel("Actual minus scheduled days")
    axes[2].set_ylabel("Number of orders")
    plt.tight_layout()
    plt.savefig("eda_overview.png", dpi=150)
    plt.close(fig)
    print("\nSaved chart: eda_overview.png")


def model_delivery(df):
    # At strict order placement, shipping_mode is excluded because it may not
    # yet have been assigned. Only use fields confirmed to be available then.
    categorical = ["order_region", "category_name", "customer_segment"]
    numerical = [
        "days_for_shipment_scheduled", "order_item_quantity",
        "order_item_discount_rate", "order_month", "order_weekday",
    ]
    features = categorical + numerical
    model_df = df.sort_values("order_date").copy()

    missing = [c for c in features if c not in model_df.columns]
    if missing:
        print("\nPredictive modelling skipped; missing feature columns:", missing)
        return

    split = int(len(model_df) * 0.8)
    train, test = model_df.iloc[:split], model_df.iloc[split:]
    if len(train) == 0 or len(test) == 0:
        print("\nPredictive modelling skipped: not enough records.")
        return
    if train["is_late"].nunique() < 2:
        print("\nClassification skipped: training data contains only one class.")
        return

    prep = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numerical),
    ])

    X_train, X_test = train[features], test[features]
    y_train, y_test = train["is_late"], test["is_late"]
    classifiers = [
        ("Logistic regression", LogisticRegression(max_iter=1000)),
        ("Random forest", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
            class_weight="balanced",
        )),
    ]
    print("\nCLASSIFICATION RESULTS (newest 20% held out)")
    for name, estimator in classifiers:
        clf = Pipeline([("prep", prep), ("model", estimator)])
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1]
        print(f"\n{name}")
        print(classification_report(y_test, pred, zero_division=0))
        if y_test.nunique() == 2:
            print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
        else:
            print("ROC-AUC unavailable: test set contains only one class.")

    # Regression is a separate target: actual shipping days.
    reg = Pipeline([("prep", prep), ("model", LinearRegression())])
    reg.fit(X_train, train["days_for_shipping_real"])
    pred_days = reg.predict(X_test)
    print("Regression MAE (days):",
          round(mean_absolute_error(test["days_for_shipping_real"], pred_days), 2))


def cluster_countries(df):
    country = df.groupby("order_country").agg(
        orders=("is_late", "size"),
        late_rate=("is_late", "mean"),
        avg_delay=("delay_days", "mean"),
        avg_sales=("sales", "mean"),
    )
    country = country[country["orders"] >= 30].dropna()
    if len(country) < 3:
        print("\nClustering skipped: fewer than three eligible countries.")
        return

    scaled = StandardScaler().fit_transform(country)
    scores = {}
    for k in range(2, min(5, len(country) - 1) + 1):
        labels = KMeans(
            n_clusters=k, n_init=10, random_state=RANDOM_STATE
        ).fit_predict(scaled)
        if len(set(labels)) > 1:
            scores[k] = silhouette_score(scaled, labels)

    if not scores:
        print("\nClustering skipped: no valid cluster count was found.")
        return
    best_k = max(scores, key=scores.get)
    country["cluster"] = KMeans(
        n_clusters=best_k, n_init=10, random_state=RANDOM_STATE
    ).fit_predict(scaled)
    print(f"\nCountry clusters (selected k={best_k}):\n",
          country.groupby("cluster").mean(numeric_only=True))


def illustrative_optimization(kpi_by_mode, total_orders=1000):
    # Placeholder assumptions only; replace with verified company inputs.
    modes = list(kpi_by_mode.index)
    late_rate = kpi_by_mode["late_rate"].to_numpy()
    cost = np.array([
        5.0 if mode == "Standard Class" else 8.0 for mode in modes
    ])
    capacity_share = np.array([
        0.6 if mode == "Same Day" else 0.9 for mode in modes
    ])
    capacity = capacity_share * total_orders
    late_penalty = 20.0

    if capacity.sum() < total_orders:
        print("\nOptimization skipped: assumed capacities cannot cover all orders.")
        return

    objective = cost + late_penalty * late_rate
    result = linprog(
        c=objective,
        A_eq=[np.ones(len(modes))],
        b_eq=[total_orders],
        bounds=[(0, capacity[i]) for i in range(len(modes))],
        method="highs",
    )
    if result.success:
        print("\nILLUSTRATIVE ALLOCATION ONLY (assumed cost/capacity):")
        print(pd.Series(
            result.x, index=modes, name="expected_orders_assigned"
        ).round(1))
        print("This is not a real recommendation; validate costs, capacities, "
              "eligibility and service constraints first.")
    else:
        print("\nNo feasible allocation found:", result.message)


def main():
    df, cancellation_rate = load_and_prepare()
    kpi_by_mode = calculate_kpis(df, cancellation_rate)
    exploratory_analysis(df, kpi_by_mode)
    model_delivery(df)
    cluster_countries(df)
    illustrative_optimization(kpi_by_mode)


if __name__ == "__main__":
    main()
