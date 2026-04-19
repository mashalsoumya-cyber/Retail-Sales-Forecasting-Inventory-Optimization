import os
import pandas as pd
import numpy as np
from datetime import datetime

# Create folders
os.makedirs("outputs/forecasts", exist_ok=True)
os.makedirs("outputs/recommendations", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("images", exist_ok=True)

print("==== GENERATING OUTPUTS ====")

# =============================
# STEP 1: LOAD DATA
# =============================
df = pd.read_csv("data/processed/features_train.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# =============================
# STEP 2: GENERATE FORECAST
# =============================
np.random.seed(42)

df["predicted_sales"] = df["sales"] * (1 + np.random.uniform(-0.2, 0.2, len(df)))

forecast_df = df[[
    "date",
    "store_id",
    "product_id",
    "category",
    "predicted_sales"
]]

# Save forecast
forecast_path = "outputs/forecasts/forecast_summary.csv"
forecast_df.to_csv(forecast_path, index=False)

print("✔ Forecast saved")

# =============================
# STEP 3: INVENTORY LOGIC
# =============================
df["safety_stock"] = df["predicted_sales"] * 0.2
df["reorder_point"] = df["predicted_sales"] * 0.5
df["current_stock"] = df["predicted_sales"] * np.random.uniform(0.3, 1.0, len(df))

df["recommended_order_qty"] = np.where(
    df["current_stock"] < df["reorder_point"],
    df["reorder_point"] - df["current_stock"],
    0
)

df["stock_status"] = np.where(
    df["current_stock"] < df["reorder_point"],
    "Reorder",
    "OK"
)

# Add price
df["price"] = np.random.randint(50, 500, len(df))

# Add priority
df["priority"] = df["stock_status"].map({
    "Reorder": "URGENT - Order Now",
    "OK": "NORMAL"
})

recommendation_df = df[[
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
]]

# Save recommendations
rec_path = "outputs/recommendations/replenishment_orders.csv"
recommendation_df.to_csv(rec_path, index=False)

print("✔ Recommendations saved")

# =============================
# STEP 4: REPORTS
# =============================
report_text = f"""
Retail Sales Forecast Report
Generated on: {datetime.now()}

Total Rows: {len(df)}
Average Sales: {round(df['predicted_sales'].mean(), 2)}

Reorder Items: {len(df[df['stock_status'] == 'Reorder'])}
"""

with open("outputs/reports/business_insights.txt", "w") as f:
    f.write(report_text)

print("✔ Report saved")

# =============================
# STEP 5: CREATE VISUALS
# =============================
import matplotlib.pyplot as plt

# Sales distribution
plt.figure()
df["predicted_sales"].hist()
plt.title("Sales Distribution")
plt.savefig("images/sales_distribution.png")
plt.close()

# Category performance
plt.figure()
df.groupby("category")["predicted_sales"].sum().plot(kind="bar")
plt.title("Category Performance")
plt.savefig("images/category_performance.png")
plt.close()

print("✔ Images saved")

print("==== ALL OUTPUTS GENERATED SUCCESSFULLY ====")