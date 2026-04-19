from __future__ import annotations

import glob
import os

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORECAST_DIR = os.path.join(BASE_DIR, "outputs", "forecasts")
RECOMMENDATION_DIR = os.path.join(BASE_DIR, "outputs", "recommendations")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")


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


st.title("📊 Analytics & Business Insights")
st.caption("Comprehensive analysis of forecasts and inventory optimization")

forecast_file = latest_valid_csv(FORECAST_DIR, "forecast_summary.csv")
recommendation_file = latest_valid_csv(RECOMMENDATION_DIR, "replenishment_orders.csv")

if not forecast_file or not recommendation_file:
    st.error("Required valid output files are missing. Please run main.py first.")
    st.stop()

try:
    forecast_df = pd.read_csv(forecast_file)
    rec_df = pd.read_csv(recommendation_file)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

if "date" in forecast_df.columns:
    forecast_df["date"] = pd.to_datetime(forecast_df["date"], errors="coerce")

if "date" in rec_df.columns:
    rec_df["date"] = pd.to_datetime(rec_df["date"], errors="coerce")

col1, col2 = st.columns(2)

avg_sales = round(float(forecast_df["predicted_sales"].mean()), 2) if "predicted_sales" in forecast_df.columns else 0
reorder_items = int((rec_df["stock_status"] == "Reorder").sum()) if "stock_status" in rec_df.columns else 0

col1.metric("Average Predicted Sales", avg_sales)
col2.metric("Reorder Items", reorder_items)

if "predicted_sales" in forecast_df.columns and "product_id" in forecast_df.columns:
    top_products = (
        forecast_df.groupby("product_id", as_index=False)["predicted_sales"]
        .sum()
        .sort_values("predicted_sales", ascending=False)
        .head(5)
    )

    st.subheader("Top 5 Products by Predicted Sales")
    st.dataframe(top_products, use_container_width=True)

if "category" in forecast_df.columns and "predicted_sales" in forecast_df.columns:
    category_summary = (
        forecast_df.groupby("category", as_index=False)["predicted_sales"]
        .sum()
        .sort_values("predicted_sales", ascending=False)
    )
    st.subheader("Category-wise Forecast")
    st.bar_chart(category_summary.set_index("category"))

business_report = os.path.join(REPORT_DIR, "business_insights.txt")
if os.path.exists(business_report):
    with open(business_report, "r", encoding="utf-8") as f:
        st.subheader("Business Notes")
        st.write(f.read())