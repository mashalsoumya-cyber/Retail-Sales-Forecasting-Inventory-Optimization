from __future__ import annotations

import glob
import os
from datetime import datetime

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORECAST_DIR = os.path.join(BASE_DIR, "outputs", "forecasts")
RECOMMENDATION_DIR = os.path.join(BASE_DIR, "outputs", "recommendations")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    page_icon="🛒",
    layout="wide"
)


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


def load_forecast_data() -> pd.DataFrame | None:
    forecast_file = latest_valid_csv(FORECAST_DIR, "forecast_summary.csv")
    if not forecast_file:
        return None

    df = pd.read_csv(forecast_file)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def load_recommendation_data() -> pd.DataFrame | None:
    recommendation_file = latest_valid_csv(RECOMMENDATION_DIR, "replenishment_orders.csv")
    if not recommendation_file:
        return None

    df = pd.read_csv(recommendation_file)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def show_dashboard(df_forecasts: pd.DataFrame, df_recommendations: pd.DataFrame) -> None:
    st.title("🛒 Retail Sales Forecasting & Inventory Optimization System")
    st.caption("AI-Powered Demand Forecasting & Smart Replenishment")

    col1, col2, col3, col4 = st.columns(4)

    avg_sales = round(float(df_forecasts["predicted_sales"].mean()), 2) if "predicted_sales" in df_forecasts.columns else 0
    reorder_items = int((df_recommendations["stock_status"] == "Reorder").sum()) if "stock_status" in df_recommendations.columns else 0

    forecast_days = 0
    if "date" in df_forecasts.columns:
        valid_dates = df_forecasts["date"].dropna()
        if len(valid_dates) > 0:
            forecast_days = (valid_dates.max() - valid_dates.min()).days + 1

    col1.metric("Forecast Rows", len(df_forecasts))
    col2.metric("Recommendation Rows", len(df_recommendations))
    col3.metric("Avg Predicted Sales", avg_sales)
    col4.metric("Forecast Period", f"{forecast_days} days")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Forecast Preview")
        st.dataframe(df_forecasts.head(10), use_container_width=True)

    with right:
        st.subheader("Recommendation Preview")
        st.dataframe(df_recommendations.head(10), use_container_width=True)

    st.divider()

    if "predicted_sales" in df_forecasts.columns and "product_id" in df_forecasts.columns:
        sales_by_product = (
            df_forecasts.groupby("product_id", as_index=False)["predicted_sales"]
            .sum()
            .sort_values("predicted_sales", ascending=False)
        )
        st.subheader("Predicted Sales by Product")
        st.bar_chart(sales_by_product.set_index("product_id"))

    report_path = os.path.join(REPORT_DIR, "business_insights.txt")
    if os.path.exists(report_path):
        st.subheader("Business Insights")
        with open(report_path, "r", encoding="utf-8") as f:
            st.info(f.read())


def main() -> None:
    df_forecasts = load_forecast_data()
    df_recommendations = load_recommendation_data()

    if df_forecasts is None or df_recommendations is None:
        st.error("No valid data files found. Please run main.py first.")
        st.stop()

    show_dashboard(df_forecasts, df_recommendations)


if __name__ == "__main__":
    main()