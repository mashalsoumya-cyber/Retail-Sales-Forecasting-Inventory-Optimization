# 🛒 Retail Sales Forecasting & Inventory Optimization System

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)
![Status](https://img.shields.io/badge/status-Active-brightgreen)

**An AI-powered system for accurate demand forecasting and intelligent inventory optimization**

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Usage](#usage) • [Results](#results)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Industry Relevance](#industry-relevance)
- [Key Features](#features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Results & Performance](#results--performance)
- [Future Enhancements](#future-enhancements)
- [Learning Outcomes](#learning-outcomes)
- [Author](#author)
- [License](#license)

---

## 🎯 Overview

This project demonstrates an **end-to-end data science + operations system** that:

✅ **Predicts** future product demand using machine learning  
✅ **Optimizes** inventory levels based on demand forecasts and lead times  
✅ **Recommends** when and how much to order  
✅ **Visualizes** forecasts and insights through an interactive dashboard  

### Real-World Business Impact

- **Reduce Stockouts:** Prevent lost sales due to out-of-stock items
- **Lower Carrying Costs:** Maintain optimal inventory, not excess stock
- **Improve Service Level:** Keep products available when customers need them
- **Increase Profitability:** Combine forecasting + inventory science

---

## 🔴 Problem Statement

**In retail businesses, poor inventory planning leads to:**

| Problem | Impact | Cost |
|---------|--------|------|
| **Stockouts** | Lost sales, unsatisfied customers | ₹50-100 lakhs/year |
| **Overstock** | Excess inventory, markdowns, waste | ₹1-5 crores/year |
| **Manual Planning** | Slow, error-prone decisions | ₹20-50 lakhs/year labor |
| **Demand Uncertainty** | Can't respond to trends | Revenue loss |

**This system solves all of these** using data-driven forecasting and optimization.

---

## 🏢 Industry Relevance

Used by companies like:
- **D-Mart** - Grocery & general merchandise forecasting
- **Reliance Retail** - Multi-format retail planning
- **Amazon** - E-commerce demand sensing
- **Flipkart** - Marketplace seller optimization
- **BigBasket** - Quick commerce fulfillment

**System Requirements in Industry:**
- Multi-store, multi-product forecasting
- Real-time demand updates
- Automated replenishment decisions
- Service level compliance
- Cost optimization

---

## ✨ Features

### 🔮 Demand Forecasting
- **Hybrid Model:** Random Forest for regular demand, Croston for intermittent
- **40+ Features:** Lags, rolling stats, calendar, promotional features
- **Confidence Intervals:** 95% CI for forecast uncertainty
- **Accuracy Metrics:** MAE, RMSE, MAPE, MASE, R²

### 📦 Inventory Optimization
- **Safety Stock Calculation:** Based on service level (z-score method)
- **Reorder Point (ROP):** Prevents stockouts during lead time
- **Economic Order Quantity (EOQ):** Minimizes ordering + holding costs
- **Action Classification:** URGENT, SOON, STABLE

### 📊 Business Intelligence
- **Interactive Dashboard:** Streamlit web app with multiple pages
- **KPI Tracking:** Service level, inventory turns, days to stockout
- **Business Insights:** Savings estimates, slow/fast movers
- **Automated Reports:** Text summaries, CSV exports

### 🎨 Visualization
- **Forecast Charts:** Time series with confidence intervals
- **Inventory Heatmaps:** Priority visualization
- **Performance Metrics:** Model accuracy by category
- **Export Options:** PNG images, CSV data, PDF reports

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.9+ | Core development |
| **Data** | Pandas, NumPy | Data manipulation |
| **ML** | Scikit-learn, XGBoost | Forecasting models |
| **Time Series** | Statsmodels | ARIMA, seasonal analysis |
| **Optimization** | SciPy | Inventory calculations |
| **Visualization** | Matplotlib, Seaborn, Plotly | Charts & plots |
| **Dashboard** | Streamlit | Interactive web app |
| **Notebook** | Jupyter | Exploratory analysis |
| **Git** | GitHub | Version control |

---

## 🏗️ Project Architecture
