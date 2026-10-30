"""
audit_raw.py -- Part A Task 2: Raw Data Audit
Audits raw orders, customers, and products without modifying them.
Compares findings against the official IIT Patna reference checkpoints.
"""
import pandas as pd
import numpy as np


def count_blanks(series: pd.Series) -> int:
    """
    Helper function to count missing values:
    Counts both pandas NaN/null values and empty or whitespace-only strings.
    """
    is_null = series.isna()
    is_whitespace = series.dropna().astype(str).str.strip() == ""
    return int(is_null.sum() + is_whitespace.sum())


def audit_tables():
    print("=" * 72)
    print("STEP 1: INSPECTING RAW DATA TABLES (SHAPE, TYPES, MISSING %)")
    print("=" * 72)

    # Load raw CSV files (read-only inspection, no transformations)
    df_customers = pd.read_csv("data/raw/customers.csv")
    df_products = pd.read_csv("data/raw/products.csv")
    df_orders = pd.read_csv("data/raw/orders.csv")

    tables = {
        "customers.csv": df_customers,
        "products.csv": df_products,
        "orders.csv": df_orders,
    }

    for name, df in tables.items():
        print(f"\n--- {name} ---")
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        summary = pd.DataFrame(
            {
                "Data Type": df.dtypes.astype(str),
                "Missing Count": df.isna().sum(),
                "Missing %": (df.isna().mean() * 100).round(2),
            }
        )
        print(summary)

    print("\n" + "=" * 72)
    print("STEP 2: COMPUTING THE 12 OFFICIAL AUDIT REFERENCE METRICS")
    print("=" * 72)

    # ----------------------------------------------------
    # ORDERS.CSV CHECKS (7 Metrics)
    # ----------------------------------------------------
    # 1. Total raw row count
    raw_orders_rows = len(df_orders)

    # 2. Number of exact duplicate rows across all columns
    exact_duplicates = int(df_orders.duplicated().sum())

    # 3. Number of rows with status == 'pending'
    pending_rows = int((df_orders["status"] == "pending").sum())

    # 4. Number of distinct order_id values
    distinct_orders = int(df_orders["order_id"].nunique())

    # 5. Non-numeric order_amount entries (text like 'N/A' or 'error' becomes NaN)
    numeric_amount = pd.to_numeric(df_orders["order_amount"], errors="coerce")
    non_numeric_amount = int(numeric_amount.isna().sum())

    # 6. Negative delivery_minutes (planted impossible durations like -10)
    numeric_deliv = pd.to_numeric(df_orders["delivery_minutes"], errors="coerce")
    negative_deliv = int((numeric_deliv < 0).sum())

    # 7. Blank delivery_minutes (both NaN and whitespace strings)
    blank_deliv = count_blanks(df_orders["delivery_minutes"])

    # ----------------------------------------------------
    # CUSTOMERS.CSV CHECKS (5 Metrics)
    # ----------------------------------------------------
    # 8. Blank age (both NaN and whitespace strings)
    blank_age = count_blanks(df_customers["age"])

    # 9. Ages outside the realistic human range [10, 100] (excluding blanks)
    numeric_age = pd.to_numeric(df_customers["age"], errors="coerce")
    invalid_age = int(((numeric_age < 10) | (numeric_age > 100)).sum())

    # 10. Distinct raw city spellings (counts messy variations like 'bengaluru', 'PUNE ')
    distinct_cities = int(df_customers["city"].nunique())

    # 11. Distinct non-blank gender spellings (e.g., 'M', 'Male', 'm', 'FEMALE')
    non_blank_genders = df_customers["gender"].dropna().loc[
        lambda s: s.astype(str).str.strip() != ""
    ]
    distinct_genders = int(non_blank_genders.nunique())

    # 12. Blank gender (both NaN and whitespace strings)
    blank_gender = count_blanks(df_customers["gender"])

    # ----------------------------------------------------
    # COMPARISON TABLE AGAINST OFFICIAL REFERENCE CHECKPOINTS
    # ----------------------------------------------------
    results = [
        ("raw orders.csv rows", raw_orders_rows, 13084),
        ("exact duplicate rows", exact_duplicates, 119),
        ("pending status rows", pending_rows, 656),
        ("distinct order_id values", distinct_orders, 12309),
        ("non-numeric order_amount entries", non_numeric_amount, 120),
        ("negative delivery_minutes", negative_deliv, 106),
        ("blank delivery_minutes", blank_deliv, 2541),
        ("customers with blank age", blank_age, 61),
        ("customers with age outside 10-100", invalid_age, 6),
        ("distinct raw city spellings", distinct_cities, 24),
        ("distinct non-blank gender spellings", distinct_genders, 10),
        ("customers with blank gender", blank_gender, 30),
    ]

    print(f"{'Audit Metric Description':<38} | {'Found':<8} | {'Reference':<10} | {'Status'}")
    print("-" * 72)

    all_matched = True
    for desc, found, ref in results:
        status = "MATCH" if found == ref else "MISMATCH"
        if status == "MISMATCH":
            all_matched = False
        print(f"{desc:<38} | {found:<8} | {ref:<10} | {status}")

    print("-" * 72)
    if all_matched:
        print(">> SUCCESS: ALL 12 AUDIT REFERENCE CHECKS MATCHED EXACTLY!")
    else:
        print(">> ATTENTION: ONE OR MORE CHECKS DID NOT MATCH.")


if __name__ == "__main__":
    audit_tables()

