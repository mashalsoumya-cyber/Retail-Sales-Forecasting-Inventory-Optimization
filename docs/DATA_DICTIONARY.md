# 📖 Data Dictionary

## Input Data Format

### Main Sales Data (`retail_sales_data.csv`)

| Column Name | Data Type | Description | Example | Required |
|-------------|-----------|-------------|---------|----------|
| store_id | Integer | Unique store identifier | 1, 2, 3 | Yes |
| item_id | Integer | Unique product/SKU identifier | 101, 102, 103 | Yes |
| date | Date (YYYY-MM-DD) | Transaction date | 2024-01-15 | Yes |
| qty_sold | Integer | Quantity sold that day | 50, 100 | Yes |
| price | Float | Price per unit on that date | 299.99, 199.50 | No |
| on_promo | Binary (0/1) | Is product on promotion | 0 or 1 | No |
| discount_pct | Float | Discount percentage | 0-100 | No |
| on_hand | Integer | Stock available at end of day | 150, 300 | No |
| unit_cost | Float | Cost to purchase one unit | 100.00, 150.00 | No |
| stockout_flag | Binary (0/1) | Did stockout occur | 0 or 1 | No |

### Data Validation Rules
