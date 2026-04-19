"""
Streamlit page for detailed forecast viewing and analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.utils import load_dataframe
from src.config import OUTPUT_FORECAST_PATH

st.set_page_config(page_title="Forecast Viewer", page_icon="🔮", layout="wide")

st.title("🔮 Detailed Forecast Viewer")
st.markdown("Explore detailed sales forecasts by store and product")

# Load forecasts
try:
    files = [f for f in os.listdir(OUTPUT_FORECAST_PATH) if f.endswith('.csv')]
    if files:
        latest_file = sorted(files)[-1]
        df_forecasts = pd.read_csv(f"{OUTPUT_FORECAST_PATH}{latest_file}")
        df_forecasts['date'] = pd.to_datetime(df_forecasts['date'])
except:
    st.error("Could not load forecast data")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

stores = sorted(df_forecasts['store_id'].unique())
selected_store = st.sidebar.selectbox("Select Store", stores)

items = sorted(df_forecasts[df_forecasts['store_id'] == selected_store]['item_id'].unique())
selected_item = st.sidebar.selectbox("Select Product", items)

# Filter data
forecast_data = df_forecasts[
    (df_forecasts['store_id'] == selected_store) &
    (df_forecasts['item_id'] == selected_item)
].sort_values('date').copy()

if len(forecast_data) == 0:
    st.error("No data available for selected combination")
    st.stop()

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Avg Forecast", f"{forecast_data['forecast'].mean():.0f} units")
with col2:
    st.metric("Max Forecast", f"{forecast_data['forecast'].max():.0f} units")
with col3:
    st.metric("Min Forecast", f"{forecast_data['forecast'].min():.0f} units")
with col4:
    st.metric("Total Forecast", f"{forecast_data['forecast'].sum():.0f} units")

st.divider()

# Main forecast chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=forecast_data['date'],
    y=forecast_data['forecast'],
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#ff7f0e', width=3),
    marker=dict(size=8),
    fill='tozeroy',
    fillcolor='rgba(255,127,14,0.2)'
))

fig.add_trace(go.Scatter(
    x=forecast_data['date'],
    y=forecast_data['upper_ci_95'],
    fill=None,
    mode='lines',
    name='Upper 95% CI',
    line=dict(color='rgba(255,127,14,0)'),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=forecast_data['date'],
    y=forecast_data['lower_ci_95'],
    fill='tonexty',
    mode='lines',
    name='Lower 95% CI',
    line=dict(color='rgba(255,127,14,0)'),
    fillcolor='rgba(100,100,255,0.1)'
))

fig.update_layout(
    title=f"Sales Forecast - Store {selected_store}, Product {selected_item}",
    xaxis_title="Date",
    yaxis_title="Quantity (units)",
    hovermode='x unified',
    height=500,
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Forecast statistics
st.subheader("📊 Forecast Statistics")

col1, col2 = st.columns(2)

with col1:
    st.metric("Forecast Days", len(forecast_data))
    st.metric("Date Range", f"{forecast_data['date'].min().date()} to {forecast_data['date'].max().date()}")

with col2:
    st.metric("Std Deviation", f"{forecast_data['forecast'].std():.2f}")
    st.metric("CV (Coefficient of Variation)", f"{forecast_data['forecast'].std()/forecast_data['forecast'].mean():.2f}")

# Data table
st.subheader("📋 Forecast Data Table")
display_df = forecast_data.copy()
display_df['date'] = display_df['date'].dt.date
st.dataframe(display_df, use_container_width=True)

# Download option
csv = forecast_data.to_csv(index=False)
st.download_button(
    label="📥 Download Forecast Data",
    data=csv,
    file_name=f"forecast_store{selected_store}_item{selected_item}_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)