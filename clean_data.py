"""
clean_data.py -- Part A Tasks 3, 4, and 5: Data Cleaning & Storage Pipeline

This script implements the exact cleaning rules defined in the official
IIT Patna capstone brief for customers.csv and orders.csv.

Outputs created:
  - data/clean/customers.csv
  - data/clean/products.csv
  - data/clean/orders.csv
  - data/instamart.db (SQLite database with customers, products, orders tables)
"""

import os
import sqlite3
import numpy as np
import pandas as pd


def clean_customers() -> pd.DataFrame:
    """
    Cleans raw customer data according to Part A Task 3:
      (a) Standardise city: strip leading/trailing spaces and convert to title case.
      (b) Standardise gender: map spellings to 'male', 'female', or 'unknown'.
      (c) Age cleaning: treat ages outside [10, 100] as missing, then impute
          all missing ages using the median age of the customer's city.
      (d) Convert signup_date to datetime.
    """
    print("\n--- Cleaning customers.csv ---")
    raw_path = os.path.join("data", "raw", "customers.csv")
    df = pd.read_csv(raw_path)

    # (a) Standardise city
    df["city"] = df["city"].astype(str).str.strip().str.title()

    # (b) Standardise gender
    def map_gender(val):
        if pd.isna(val):
            return "unknown"
        s = str(val).strip().lower()
        if s in ["m", "male"]:
            return "male"
        elif s in ["f", "female"]:
            return "female"
        return "unknown"

    df["gender"] = df["gender"].apply(map_gender)

    # (c) Clean and impute age
    # Convert to numeric, setting non-numeric or blanks to NaN
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    # Mark ages outside 10-100 as missing
    df.loc[(df["age"] < 10) | (df["age"] > 100), "age"] = np.nan
    # Impute missing ages with city median
    city_median_age = df.groupby("city")["age"].transform("median")
    df["age"] = df["age"].fillna(city_median_age).round().astype(int)

    # (d) Convert signup_date to datetime
    df["signup_date"] = pd.to_datetime(df["signup_date"])

    print(f"Cleaned customers shape: {df.shape}")
    return df


def clean_orders(df_products: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw order data according to Part A Task 4 in strict order:
      Rule 1: Convert order_ts to datetime.
      Rule 2: Remove exact duplicate rows.
      Rule 3: Keep only the latest record for each order_id.
      Rule 4: Convert order_amount to numeric and fill missing values using
              price_inr * quantity * (1 - discount_pct / 100).
      Rule 5: Set negative delivery_minutes to missing. For completed orders,
              impute missing delivery_minutes with product category median.
              For cancelled orders, leave delivery_minutes as missing.
    """
    print("\n--- Cleaning orders.csv ---")
    raw_path = os.path.join("data", "raw", "orders.csv")
    df = pd.read_csv(raw_path)

    # Rule 1: Convert order_ts to datetime
    df["order_ts"] = pd.to_datetime(df["order_ts"])

    # Rule 2: Remove exact duplicate rows
    dups_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Rule 2: Dropped {dups_count} exact duplicate rows.")

    # Rule 3: Keep record with latest order_ts per order_id
    # Sort chronologically by order_ts and keep the last record per order_id
    df = df.sort_values("order_ts").drop_duplicates(subset=["order_id"], keep="last")
    # Sort cleanly by order_id and reset index
    df = df.sort_values("order_id").reset_index(drop=True)
    print(f"Rule 3: Deduplicated by order_id (kept latest status). Remaining rows: {len(df)}")

    # Rule 4: Fix order_amount using product price lookup
    price_map = df_products.set_index("product_id")["price_inr"].to_dict()
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")
    missing_amount_mask = df["order_amount"].isna()
    missing_amount_count = missing_amount_mask.sum()

    # Calculate expected amount: round(price_inr * quantity * (1 - discount_pct / 100), 2)
    calculated_amount = (
        df["product_id"].map(price_map)
        * df["quantity"]
        * (1 - df["discount_pct"] / 100.0)
    ).round(2)
    df.loc[missing_amount_mask, "order_amount"] = calculated_amount.loc[missing_amount_mask]
    print(f"Rule 4: Imputed {missing_amount_count} missing/corrupted order_amount entries.")

    # Rule 5: Fix delivery_minutes
    df["delivery_minutes"] = pd.to_numeric(df["delivery_minutes"], errors="coerce")
    # Set negative delivery_minutes to NaN
    neg_mask = df["delivery_minutes"] < 0
    neg_count = neg_mask.sum()
    df.loc[neg_mask, "delivery_minutes"] = np.nan
    print(f"Rule 5: Set {neg_count} negative delivery times to missing.")

    # Map product category to compute category-level delivery medians
    cat_map = df_products.set_index("product_id")["category"].to_dict()
    df["category"] = df["product_id"].map(cat_map)

    # Impute missing delivery_minutes for completed orders only
    completed_mask = df["status"] == "completed"
    category_delivery_medians = (
        df[completed_mask].groupby("category")["delivery_minutes"].transform("median")
    )
    df.loc[completed_mask, "delivery_minutes"] = (
        df.loc[completed_mask, "delivery_minutes"].fillna(category_delivery_medians)
    )

    # Cancelled orders remain NaN (as required by official brief)
    # Drop the temporary category column to preserve original orders schema
    df = df.drop(columns=["category"])
    print("Rule 5: Imputed completed order delivery times; left cancelled orders as missing.")

    return df


def validate_clean_data(
    df_customers: pd.DataFrame, df_products: pd.DataFrame, df_orders: pd.DataFrame
):
    """
    Validates the cleaned DataFrames against all reference checkpoints
    given in Part A Tasks 3 and 4 of the official brief.
    """
    print("\n" + "=" * 70)
    print("VALIDATING CLEANED DATA AGAINST OFFICIAL REFERENCE VALUES")
    print("=" * 70)

    # --- Customers Checks ---
    print("\n[Customers Validation]")
    assert df_customers.shape == (1200, 5), f"Customers shape mismatch: {df_customers.shape} != (1200, 5)"
    cust_missing_total = int(df_customers.isna().sum().sum())
    assert cust_missing_total == 0, f"Customers has {cust_missing_total} missing values (expected 0)!"
    print(f">> Customers shape and completeness: PASS (shape={df_customers.shape}, missing={cust_missing_total})")

    city_counts = df_customers["city"].value_counts().to_dict()
    ref_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }
    print(f"City distribution: {city_counts}")
    assert city_counts == ref_cities, f"City counts mismatch! Found: {city_counts}"
    print(">> City distribution: PASS (Pune: 227, Delhi: 210, Bengaluru: 197, Chennai: 191, Hyderabad: 191, Mumbai: 184)")

    gender_counts = df_customers["gender"].value_counts().to_dict()
    ref_genders = {"female": 610, "male": 560, "unknown": 30}
    print(f"Gender distribution: {gender_counts}")
    assert gender_counts == ref_genders, f"Gender counts mismatch! Found: {gender_counts}"
    print(">> Gender distribution: PASS (female: 610, male: 560, unknown: 30)")

    missing_ages = df_customers["age"].isna().sum()
    min_age, max_age = df_customers["age"].min(), df_customers["age"].max()
    print(f"Missing ages: {missing_ages}, Age range: [{min_age}, {max_age}]")
    assert missing_ages == 0, "Missing ages must be 0!"
    assert 18 <= min_age and max_age <= 60, f"Ages outside 18-60: [{min_age}, {max_age}]"
    print(">> Age checks: PASS (0 missing ages, all between 18 and 60)")

    # --- Products Checks ---
    print("\n[Products Validation]")
    assert df_products.shape == (40, 4), f"Products shape mismatch: {df_products.shape} != (40, 4)"
    prod_missing_total = int(df_products.isna().sum().sum())
    assert prod_missing_total == 0, f"Products has {prod_missing_total} missing values (expected 0)!"
    print(f">> Products shape and completeness: PASS (shape={df_products.shape}, missing={prod_missing_total})")

    # --- Orders Checks ---
    print("\n[Orders Validation]")
    total_orders = len(df_orders)
    distinct_orders = df_orders["order_id"].nunique()
    assert total_orders == 12309, f"Orders count mismatch: {total_orders} != 12309"
    assert distinct_orders == 12309, f"Distinct order_id mismatch: {distinct_orders} != 12309"
    print(f">> Total orders: PASS ({total_orders} rows, one per order_id)")

    status_counts = df_orders["status"].value_counts().to_dict()
    print(f"Status distribution: {status_counts}")
    assert status_counts.get("completed", 0) == 11110, "Completed count mismatch!"
    assert status_counts.get("cancelled", 0) == 1199, "Cancelled count mismatch!"
    assert status_counts.get("pending", 0) == 0, "Pending count must be 0!"
    print(">> Order status distribution: PASS (completed: 11,110, cancelled: 1,199, pending: 0)")

    missing_amounts = df_orders["order_amount"].isna().sum()
    assert missing_amounts == 0, f"Missing order_amount: {missing_amounts} != 0"
    print(">> Missing order_amount: PASS (0 missing)")

    completed_orders = df_orders[df_orders["status"] == "completed"]
    cancelled_orders = df_orders[df_orders["status"] == "cancelled"]

    missing_deliv_completed = completed_orders["delivery_minutes"].isna().sum()
    missing_deliv_cancelled = cancelled_orders["delivery_minutes"].isna().sum()
    assert missing_deliv_completed == 0, "Missing delivery_minutes for completed must be 0!"
    assert missing_deliv_cancelled == 1199, "Missing delivery_minutes for cancelled must be 1199!"
    print(">> Delivery minutes missingness: PASS (0 missing completed, 1,199 missing cancelled)")

    revenue_completed = round(completed_orders["order_amount"].sum(), 2)
    print(f"Total completed revenue: INR {revenue_completed:,.2f}")
    assert revenue_completed == 2998668.80, f"Revenue mismatch: {revenue_completed} != 2998668.80"
    print(">> Completed revenue sum: PASS (INR 2,998,668.80)")

    print("\n" + "=" * 70)
    print("ALL DATA CLEANING CHECKPOINTS PASSED PERFECTLY!")
    print("=" * 70)


def save_and_store(
    df_customers: pd.DataFrame, df_products: pd.DataFrame, df_orders: pd.DataFrame
):
    """
    Saves clean CSV files to data/clean/ and exports tables to data/instamart.db.
    """
    clean_dir = os.path.join("data", "clean")
    os.makedirs(clean_dir, exist_ok=True)

    # 1. Save cleaned CSVs (dates formatted cleanly)
    customers_csv_path = os.path.join(clean_dir, "customers.csv")
    products_csv_path = os.path.join(clean_dir, "products.csv")
    orders_csv_path = os.path.join(clean_dir, "orders.csv")

    df_customers.to_csv(customers_csv_path, index=False)
    df_products.to_csv(products_csv_path, index=False)
    df_orders.to_csv(orders_csv_path, index=False)
    print(f"\nSaved clean CSV files to: {clean_dir}/")

    # 2. Store tables in SQLite database data/instamart.db
    db_path = os.path.join("data", "instamart.db")
    conn = sqlite3.connect(db_path)

    # Ensure timestamps are stored as text 'YYYY-MM-DD HH:MM:SS' for SQLite compatibility
    df_orders_db = df_orders.copy()
    df_orders_db["order_ts"] = df_orders_db["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df_customers_db = df_customers.copy()
    df_customers_db["signup_date"] = df_customers_db["signup_date"].dt.strftime("%Y-%m-%d")

    df_customers_db.to_sql("customers", conn, if_exists="replace", index=False)
    df_products.to_sql("products", conn, if_exists="replace", index=False)
    df_orders_db.to_sql("orders", conn, if_exists="replace", index=False)

    conn.close()
    print(f"Stored clean tables ('customers', 'products', 'orders') in: {db_path}")


def main():
    # 1. Load products (read-only reference)
    df_products = pd.read_csv(os.path.join("data", "raw", "products.csv"))

    # 2. Clean customers and orders
    df_customers = clean_customers()
    df_orders = clean_orders(df_products)

    # 3. Validate against official checkpoints
    validate_clean_data(df_customers, df_products, df_orders)

    # 4. Save to data/clean/ and data/instamart.db
    save_and_store(df_customers, df_products, df_orders)


if __name__ == "__main__":
    main()

