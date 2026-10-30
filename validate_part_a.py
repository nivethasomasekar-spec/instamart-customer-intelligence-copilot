"""
validate_part_a.py -- Part A Task 5: Independent Verification Script

Independently validates the cleaned CSV files in data/clean/ and the SQLite
database data/instamart.db against the official reference checkpoints.

When all checks pass, this script prints exactly:
ALL CHECKPOINTS PASSED
"""

import os
import sqlite3
import pandas as pd


def validate_clean_csvs():
    print("--- 1. Validating Cleaned CSV Files (data/clean/) ---")

    # Load cleaned CSVs
    cust_path = os.path.join("data", "clean", "customers.csv")
    prod_path = os.path.join("data", "clean", "products.csv")
    ord_path = os.path.join("data", "clean", "orders.csv")

    assert os.path.exists(cust_path), f"Missing {cust_path}"
    assert os.path.exists(prod_path), f"Missing {prod_path}"
    assert os.path.exists(ord_path), f"Missing {ord_path}"

    df_cust = pd.read_csv(cust_path)
    df_prod = pd.read_csv(prod_path)
    df_ord = pd.read_csv(ord_path)

    # Check 1: Shapes
    assert df_cust.shape == (1200, 5), f"Customers shape mismatch: {df_cust.shape} != (1200, 5)"
    assert df_prod.shape == (40, 4), f"Products shape mismatch: {df_prod.shape} != (40, 4)"
    assert df_ord.shape == (12309, 9), f"Orders shape mismatch: {df_ord.shape} != (12309, 9)"
    print("Checkpoint 1/8 [Shapes]: PASS")

    # Check 2: Customer completeness & age bounds
    assert df_cust.isna().sum().sum() == 0, "Customers table has missing values!"
    assert df_prod.isna().sum().sum() == 0, "Products table has missing values!"
    assert (df_cust["age"] >= 18).all() and (df_cust["age"] <= 60).all(), "Ages outside [18, 60] found!"
    print("Checkpoint 2/8 [Customer & Product Completeness]: PASS")

    # Check 3: City distribution
    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }
    assert df_cust["city"].value_counts().to_dict() == expected_cities, "City counts mismatch!"
    print("Checkpoint 3/8 [City Distribution]: PASS")

    # Check 4: Gender distribution
    expected_genders = {"female": 610, "male": 560, "unknown": 30}
    assert df_cust["gender"].value_counts().to_dict() == expected_genders, "Gender counts mismatch!"
    print("Checkpoint 4/8 [Gender Distribution]: PASS")

    # Check 5: Order status counts and unique order IDs
    assert df_ord["order_id"].nunique() == 12309, "Distinct order_id count != 12309"
    status_counts = df_ord["status"].value_counts().to_dict()
    assert status_counts.get("completed", 0) == 11110, "Completed orders != 11,110"
    assert status_counts.get("cancelled", 0) == 1199, "Cancelled orders != 1,199"
    assert status_counts.get("pending", 0) == 0, "Pending orders != 0"
    print("Checkpoint 5/8 [Order Status & ID Uniqueness]: PASS")

    # Check 6: Orders missingness requirements
    assert df_ord["order_amount"].isna().sum() == 0, "Missing order_amount values found!"
    completed_orders = df_ord[df_ord["status"] == "completed"]
    cancelled_orders = df_ord[df_ord["status"] == "cancelled"]
    assert completed_orders["delivery_minutes"].isna().sum() == 0, "Missing delivery_minutes in completed orders!"
    assert cancelled_orders["delivery_minutes"].isna().sum() == 1199, "Cancelled orders missing delivery != 1,199!"
    print("Checkpoint 6/8 [Orders Missingness Rules]: PASS")

    # Check 7: Completed revenue sum
    completed_revenue = round(completed_orders["order_amount"].sum(), 2)
    assert completed_revenue == 2998668.80, f"Completed revenue mismatch: {completed_revenue} != 2998668.80"
    print(f"Checkpoint 7/8 [Completed Revenue = INR 2,998,668.80]: PASS")


def validate_sqlite_database():
    print("\n--- 2. Validating SQLite Database (data/instamart.db) ---")

    db_path = os.path.join("data", "instamart.db")
    assert os.path.exists(db_path), f"Missing database file at {db_path}"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Verify tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    required_tables = {"customers", "products", "orders"}
    assert required_tables.issubset(tables), f"Database missing tables: {required_tables - tables}"

    # Verify row counts in SQLite tables
    cursor.execute("SELECT COUNT(*) FROM customers;")
    cust_count = cursor.fetchone()[0]
    assert cust_count == 1200, f"SQLite customers row count mismatch: {cust_count} != 1200"

    cursor.execute("SELECT COUNT(*) FROM products;")
    prod_count = cursor.fetchone()[0]
    assert prod_count == 40, f"SQLite products row count mismatch: {prod_count} != 40"

    cursor.execute("SELECT COUNT(*) FROM orders;")
    ord_count = cursor.fetchone()[0]
    assert ord_count == 12309, f"SQLite orders row count mismatch: {ord_count} != 12309"

    # Verify order_ts text format in SQLite
    cursor.execute("SELECT order_ts FROM orders LIMIT 1;")
    sample_ts = cursor.fetchone()[0]
    assert isinstance(sample_ts, str) and len(sample_ts) == 19, f"Invalid order_ts format: {sample_ts}"

    conn.close()
    print("Checkpoint 8/8 [SQLite Database & Table Row Counts]: PASS")


def main():
    validate_clean_csvs()
    validate_sqlite_database()
    # The official brief strictly requires printing exactly this line:
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()

