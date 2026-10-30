"""
eda_analysis.py -- Part A Task 7: Pandas Cross-Check and Exploratory Data Analysis

Performs the pandas cross-check of SQL Q2, computes city-level named aggregations,
analyzes the order_amount distribution (mean, median, skew, IQR, fences, outliers),
calculates the correlation matrix, and outputs the four-question EDA interpretation.
"""

import os
import numpy as np
import pandas as pd


def run_eda():
    print("=" * 75)
    print("PART A TASK 7: PANDAS CROSS-CHECK AND EXPLORATORY DATA ANALYSIS")
    print("=" * 75)

    # 1. Load cleaned datasets
    cust_path = os.path.join("data", "clean", "customers.csv")
    prod_path = os.path.join("data", "clean", "products.csv")
    ord_path = os.path.join("data", "clean", "orders.csv")

    df_cust = pd.read_csv(cust_path)
    df_prod = pd.read_csv(prod_path)
    df_ord = pd.read_csv(ord_path)

    # Filter for completed orders
    completed_orders = df_ord[df_ord["status"] == "completed"].copy()
    print(f"Loaded completed orders: {len(completed_orders)} rows")

    # Merge tables to mirror SQL Q2
    merged = completed_orders.merge(
        df_cust[["customer_id", "city"]], on="customer_id"
    ).merge(df_prod[["product_id", "category"]], on="product_id")

    # -------------------------------------------------------------
    # 2 & 3. Rebuild Q2 via pivot_table(margins=True) and Cross-Check
    # -------------------------------------------------------------
    print("\n--- 1. Pivot Table Cross-Check of SQL Q2 ---")
    pivot_q2 = pd.pivot_table(
        merged,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
    ).round(2)

    print("Pivot Table (City x Category Revenue with Margins):")
    print(pivot_q2)

    grand_total = round(pivot_q2.loc["All", "All"], 2)
    print(f"\nPivot Table Grand Total: INR {grand_total:,.2f}")
    assert grand_total == 2998668.80, f"Grand total mismatch: {grand_total} != 2998668.80"
    print(">> Checkpoint [Pivot Grand Total == SQL Total]: PASS (INR 2,998,668.80)")

    # -------------------------------------------------------------
    # 4. Per-City Named Aggregation
    # -------------------------------------------------------------
    print("\n--- 2. Per-City Named Aggregations (Sum, Mean, Count) ---")
    city_summary = (
        merged.groupby("city")["order_amount"]
        .agg(
            total_spend="sum",
            mean_spend="mean",
            order_count="count",
        )
        .round(2)
    )
    print(city_summary)

    # Check Bengaluru
    blr = city_summary.loc["Bengaluru"]
    print(f"\nBengaluru: sum={blr['total_spend']}, mean={blr['mean_spend']}, count={int(blr['order_count'])}")
    assert blr["total_spend"] == 523643.40, f"Bengaluru sum mismatch: {blr['total_spend']}"
    assert blr["mean_spend"] == 276.47, f"Bengaluru mean mismatch: {blr['mean_spend']}"
    assert blr["order_count"] == 1894, f"Bengaluru count mismatch: {blr['order_count']}"
    print(">> Checkpoint [Bengaluru Aggregations]: PASS (Sum: 523,643.40, Mean: 276.47, Count: 1,894)")

    # Check Mumbai
    mum = city_summary.loc["Mumbai"]
    print(f"Mumbai: sum={mum['total_spend']}, mean={mum['mean_spend']}, count={int(mum['order_count'])}")
    assert mum["total_spend"] == 479058.50, f"Mumbai sum mismatch: {mum['total_spend']}"
    assert mum["mean_spend"] == 297.00, f"Mumbai mean mismatch: {mum['mean_spend']}"
    assert mum["order_count"] == 1613, f"Mumbai count mismatch: {mum['order_count']}"
    print(">> Checkpoint [Mumbai Aggregations]: PASS (Sum: 479,058.50, Mean: 297.00, Count: 1,613)")

    # -------------------------------------------------------------
    # 5. Distribution Statistics & 1.5xIQR Outlier Analysis
    # -------------------------------------------------------------
    print("\n--- 3. Order Amount Distribution & Outlier Detection ---")
    amounts = completed_orders["order_amount"]
    mean_val = round(amounts.mean(), 2)
    median_val = round(amounts.median(), 2)
    skew_val = round(amounts.skew(), 2)
    q1 = round(amounts.quantile(0.25), 2)
    q3 = round(amounts.quantile(0.75), 2)
    iqr = round(q3 - q1, 2)
    lower_fence = round(q1 - 1.5 * iqr, 2)
    upper_fence = round(q3 + 1.5 * iqr, 2)

    outliers_below = int((amounts < lower_fence).sum())
    outliers_above = int((amounts > upper_fence).sum())

    print(f"Mean:         INR {mean_val}")
    print(f"Median:       INR {median_val}")
    print(f"Skewness:     {skew_val}")
    print(f"Q1 (25%):     INR {q1}")
    print(f"Q3 (75%):     INR {q3}")
    print(f"IQR:          INR {iqr}")
    print(f"Lower Fence:  INR {lower_fence} (Outliers below: {outliers_below})")
    print(f"Upper Fence:  INR {upper_fence} (Outliers above: {outliers_above})")

    assert mean_val == 269.91, f"Mean mismatch: {mean_val} != 269.91"
    assert median_val == 135.00, f"Median mismatch: {median_val} != 135.00"
    assert abs(skew_val - 4.61) <= 0.05, f"Skewness mismatch: {skew_val} != 4.61"
    assert q1 == 64.80, f"Q1 mismatch: {q1} != 64.80"
    assert q3 == 297.00, f"Q3 mismatch: {q3} != 297.00"
    assert upper_fence == 645.30, f"Upper fence mismatch: {upper_fence} != 645.30"
    assert outliers_above == 897, f"Outliers above mismatch: {outliers_above} != 897"
    assert outliers_below == 0, f"Outliers below mismatch: {outliers_below} != 0"
    print(">> Checkpoint [Distribution & IQR Outliers]: PASS (Mean: 269.91, Median: 135.00, Skew: 4.61, Upper Fence: 645.30, Outliers: 897)")

    # -------------------------------------------------------------
    # 6 & 7. Correlation Matrix & Strongest Pair
    # -------------------------------------------------------------
    print("\n--- 4. Correlation Matrix of Numerical Order Variables ---")
    num_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr_matrix = completed_orders[num_cols].corr().round(3)
    print(corr_matrix)

    # Extract correlation pairs excluding diagonal
    corr_unstacked = corr_matrix.where(
        ~np.eye(corr_matrix.shape[0], dtype=bool)
    ).unstack().dropna()

    abs_corr = corr_unstacked.abs()
    top_pair = abs_corr.idxmax()
    top_val = corr_unstacked[top_pair]
    print(f"\nStrongest linear correlation pair: {top_pair} with r = {top_val:.3f}")

    assert set(top_pair) == {"order_amount", "discount_pct"}, f"Top pair mismatch: {top_pair}"
    assert abs(top_val - (-0.095)) <= 0.01, f"Top correlation value mismatch: {top_val}"

    # Also check the second pair mentioned in brief (order_amount and quantity ~ +0.087)
    amt_qty_corr = corr_matrix.loc["order_amount", "quantity"]
    print(f"Second strongest correlation pair: ('order_amount', 'quantity') with r = {amt_qty_corr:.3f}")
    assert abs(amt_qty_corr - 0.087) <= 0.01, f"order_amount/quantity mismatch: {amt_qty_corr}"
    print(">> Checkpoint [Correlation Analysis]: PASS (Top pair: order_amount / discount_pct ~ -0.095, second: +0.087)")

    # -------------------------------------------------------------
    # 8. Four-Question EDA Business Interpretation (<= 200 words)
    # -------------------------------------------------------------
    print("\n--- 5. Four-Question EDA Business Interpretation ---")
    interpretation = (
        "1. What is this data about?\n"
        "It captures 11,110 completed Swiggy Instamart grocery orders placed across 6 metro cities "
        "between April and September 2025.\n\n"
        "2. What do its distributions and statistics look like?\n"
        "The distribution is heavily right-skewed (skewness 4.61). The mean order value is INR 269.91, "
        "while the median is just INR 135.00, demonstrating that the vast majority of orders are small grocery baskets.\n\n"
        "3. Are there outliers or anomalies?\n"
        "Yes. With Q1 at INR 64.80 and Q3 at INR 297.00 (IQR = INR 232.20), the upper 1.5xIQR fence is INR 645.30. "
        "There are exactly 897 high-value outlier orders above this threshold (mainly electronics like earbuds and power banks) "
        "and zero outliers below the lower fence.\n\n"
        "4. What relationships exist between variables?\n"
        "Linear correlations among order metrics are very weak. The largest correlation in absolute value is between "
        "order_amount and discount_pct at r ≈ -0.095, followed by order_amount and quantity at r ≈ +0.087. "
        "Crucially, a correlation of this negligible magnitude does not establish causation."
    )
    print(interpretation)
    word_count = len(interpretation.split())
    print(f"\n[Interpretation Word Count: {word_count} words (Limit: at most 200 words)]")
    assert word_count <= 200, f"Interpretation exceeds 200 words: {word_count}"

    print("\n" + "=" * 75)
    print("ALL PART A TASK 7 EDA CHECKPOINTS PASSED PERFECTLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_eda()

