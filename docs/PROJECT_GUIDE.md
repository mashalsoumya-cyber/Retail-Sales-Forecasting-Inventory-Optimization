# 📚 Complete Project Guide - Retail Sales Forecasting & Inventory Optimization

## Table of Contents
1. [Overview](#overview)
2. [Setup Instructions](#setup-instructions)
3. [Project Structure](#project-structure)
4. [How to Run](#how-to-run)
5. [Data Format](#data-format)
6. [Model Explanation](#model-explanation)
7. [Inventory Calculations](#inventory-calculations)
8. [Dashboard Usage](#dashboard-usage)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This system combines **demand forecasting** and **inventory optimization** to help retailers:
- Predict future sales accurately
- Optimize stock levels
- Reduce stockouts and overstock situations
- Improve profitability

### Key Components

| Component | Purpose |
|-----------|---------|
| **Data Loader** | Ingest and validate sales data |
| **Preprocessor** | Clean and prepare data |
| **EDA** | Understand trends and patterns |
| **Feature Engineer** | Create ML features |
| **Forecaster** | Predict future demand |
| **Optimizer** | Calculate optimal inventory levels |
| **Dashboard** | Visualize results and recommendations |

---

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- pip package manager
- 2GB free disk space

### Installation Steps

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python test_installation.py