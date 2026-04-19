from __future__ import annotations

import glob
import os

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECOMMENDATION_DIR = os.path.join(BASE_DIR, "outputs", "recommendations")


def is_valid_csv(file_path: str) -> bool:
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
        df = pd.read_csv(file_path, nrows=1)
        return len(df.columns) > 0
    except Exception:
        return False


def latest_valid_csv(folder_path: str, fallback_name: str | None = None) -> str | None:
    if fallback_name:
        fallback_path = os.path.join(folder_path, fallback_name)
        if is_valid_csv(fallback_path):
            return fallback_path

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    csv_files = sorted(csv_files, key=os.path.getmtime, reverse=True)

    for file_path in csv_files:
        if is_valid_csv(file_path):
            return file_path

    return None


def add_priority_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "priority" not in df.columns:
        if "stock_status" in df.columns:
            df["priority"] = df["stock_status"].map({
                "Reorder": "URGENT - Order Now",
                "Monitor": "SOON - Plan Order"
            }).fillna("NORMAL")
        else:
            df["priority"] = "NORMAL"

    return df


st.title("📦 Inventory Optimization & Replenishment")
st.caption("Smart replenishment recommendations based on demand forecasts")

recommendation_file = latest_valid_csv(RECOMMENDATION_DIR, "replenishment_orders.csv")

if not recommendation_file:
    st.error("Could not load recommendation data. Please run main.py again.")
    st.stop()

try:
    rec_df = pd.read_csv(recommendation_file)
except Exception as e:
    st.error(f"Could not load recommendation data: {e}")
    st.stop()

if "date" in rec_df.columns:
    rec_df["date"] = pd.to_datetime(rec_df["date"], errors="coerce")

rec_df = add_priority_column(rec_df)

st.sidebar.header("Filters")

store_options = sorted(rec_df["store_id"].dropna().astype(str).unique().tolist()) if "store_id" in rec_df.columns else []
selected_store = st.sidebar.selectbox("Select Store", store_options) if store_options else "All"

priority_options = sorted(rec_df["priority"].dropna().unique().tolist())
selected_priority = st.sidebar.multiselect(
    "Filter by Priority",
    options=priority_options,
    default=priority_options
)

filtered_recs = rec_df.copy()

if "store_id" in filtered_recs.columns and selected_store != "All":
    filtered_recs = filtered_recs[filtered_recs["store_id"].astype(str) == str(selected_store)]

if "priority" in filtered_recs.columns and selected_priority:
    filtered_recs = filtered_recs[filtered_recs["priority"].isin(selected_priority)]

col1, col2, col3, col4 = st.columns(4)

total_orders = len(filtered_recs)
units_to_order = filtered_recs["recommended_order_qty"].sum() if "recommended_order_qty" in filtered_recs.columns else 0

budget_required = 0
if "recommended_order_qty" in filtered_recs.columns and "price" in filtered_recs.columns:
    budget_required = (filtered_recs["recommended_order_qty"] * filtered_recs["price"]).sum()

avg_days_to_stock = 2

col1.metric("Total Orders", int(total_orders))
col2.metric("Units to Order", int(units_to_order))
col3.metric("Budget Required", f"₹{int(budget_required)}")
col4.metric("Avg Days to Stock", avg_days_to_stock)

st.divider()
st.subheader("📋 Replenishment Orders")

display_cols = [
    col for col in [
        "date",
        "store_id",
        "product_id",
        "category",
        "predicted_sales",
        "safety_stock",
        "reorder_point",
        "recommended_order_qty",
        "stock_status",
        "priority",
        "price"
    ] if col in filtered_recs.columns
]

if "priority" in filtered_recs.columns:
    priority_order = {
        "URGENT - Order Now": 0,
        "SOON - Plan Order": 1,
        "NORMAL": 2
    }
    filtered_recs["priority_sort"] = filtered_recs["priority"].map(priority_order).fillna(99)
    filtered_recs = filtered_recs.sort_values(["priority_sort"])
    filtered_recs = filtered_recs.drop(columns=["priority_sort"])

st.dataframe(filtered_recs[display_cols], use_container_width=True)

if "priority" in filtered_recs.columns:
    st.subheader("Priority Distribution")
    priority_counts = filtered_recs["priority"].value_counts().rename_axis("priority").reset_index(name="count")
    st.bar_chart(priority_counts.set_index("priority"))