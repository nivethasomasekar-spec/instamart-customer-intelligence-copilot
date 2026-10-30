"""
run_sql_analysis.py -- Part A Task 6: SQL Analysis & Verification Script

Executes all 7 analytical SQL queries from sql/queries.sql against data/instamart.db
using pandas.read_sql_query and rigorously validates all reference checkpoints.
"""

import os
import sqlite3
import pandas as pd


def load_queries(sql_file_path: str) -> dict:
    """
    Parses sql/queries.sql and extracts individual queries by their [Q1]..[Q7] tags.
    """
    with open(sql_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    queries = {}
    for i in range(1, 8):
        tag = f"-- [Q{i}]"
        next_tag = f"-- [Q{i+1}]" if i < 7 else None
        
        start = content.find(tag)
        assert start != -1, f"Could not find query tag {tag} in {sql_file_path}"
        start += len(tag)
        
        if next_tag:
            end = content.find(next_tag)
            assert end != -1, f"Could not find next tag {next_tag}"
            q_text = content[start:end].strip()
        else:
            q_text = content[start:].strip()
            
        queries[f"Q{i}"] = q_text
    return queries


def run_and_validate():
    db_path = os.path.join("data", "instamart.db")
    sql_path = os.path.join("sql", "queries.sql")

    assert os.path.exists(db_path), f"Missing database: {db_path}"
    assert os.path.exists(sql_path), f"Missing SQL file: {sql_path}"

    conn = sqlite3.connect(db_path)
    queries = load_queries(sql_path)

    print("=" * 75)
    print("PART A TASK 6: EXECUTING SQL QUERIES AGAINST data/instamart.db")
    print("=" * 75)

    # -------------------------------------------------------------
    # Q1: WHERE then HAVING
    # -------------------------------------------------------------
    print("\n[Q1: WHERE order_amount > 500 then HAVING SUM(order_amount) > 8000]")
    df_q1 = pd.read_sql_query(queries["Q1"], conn)
    print(f"Returned customers: {len(df_q1)}")
    print("Top 3 rows:\n", df_q1.head(3).to_string(index=False))
    assert len(df_q1) == 30, f"Q1 expected 30 customers, got {len(df_q1)}"
    print(">> Q1 Checkpoint: PASS (30 customers)")

    # -------------------------------------------------------------
    # Q2: Three-Table JOIN (City x Category Revenue)
    # -------------------------------------------------------------
    print("\n[Q2: Revenue by City and Product Category (3-Table JOIN)]")
    df_q2 = pd.read_sql_query(queries["Q2"], conn)
    print(f"Returned rows: {len(df_q2)}")
    total_rev_q2 = round(df_q2["revenue"].sum(), 2)
    print(f"Total Revenue across all cells: INR {total_rev_q2:,.2f}")
    assert len(df_q2) == 30, f"Q2 expected 30 rows, got {len(df_q2)}"
    assert total_rev_q2 == 2998668.80, f"Q2 revenue mismatch: {total_rev_q2} != 2998668.80"

    # Largest cell verification
    largest_cell = df_q2.loc[df_q2["revenue"].idxmax()]
    print(f"Largest cell: {largest_cell['city']} / {largest_cell['category']} = INR {largest_cell['revenue']:,.2f}")
    assert largest_cell["city"] == "Bengaluru", f"Largest city: {largest_cell['city']} != Bengaluru"
    assert largest_cell["category"] == "Electronics", f"Largest category: {largest_cell['category']} != Electronics"
    assert round(largest_cell["revenue"], 2) == 233909.45, f"Largest revenue mismatch: {largest_cell['revenue']} != 233909.45"
    print(">> Q2 Checkpoints: PASS (30 rows, total INR 2,998,668.80, max Bengaluru/Electronics INR 233,909.45)")

    # -------------------------------------------------------------
    # Q3: Scalar Subquery (Above Average Catalogue Price)
    # -------------------------------------------------------------
    print("\n[Q3: Products Priced Above Average Catalogue Price (Scalar Subquery)]")
    df_q3 = pd.read_sql_query(queries["Q3"], conn)
    print(f"Returned products: {len(df_q3)}")
    print("Top 3 products:\n", df_q3.head(3).to_string(index=False))
    assert len(df_q3) == 11, f"Q3 expected 11 products, got {len(df_q3)}"
    print(">> Q3 Checkpoint: PASS (11 products)")

    # -------------------------------------------------------------
    # Q4: LEFT JOIN (Customers with No Completed Orders)
    # -------------------------------------------------------------
    print("\n[Q4: Customers with No Completed Orders (LEFT JOIN)]")
    df_q4 = pd.read_sql_query(queries["Q4"], conn)
    print(f"Returned customers: {len(df_q4)}")
    print("Sample:\n", df_q4.head(3).to_string(index=False))
    assert len(df_q4) == 25, f"Q4 expected 25 customers, got {len(df_q4)}"
    print(">> Q4 Checkpoint: PASS (25 customers)")

    # -------------------------------------------------------------
    # Q5: Common Table Expressions (Above Average Spenders)
    # -------------------------------------------------------------
    print("\n[Q5: Customers Spending Above Average Total Spend (CTEs)]")
    # Also verify the exact average computed in the CTE
    avg_spend_sql = """
        WITH customer_spend AS (
            SELECT customer_id, SUM(order_amount) AS total_spend
            FROM orders WHERE status = 'completed' GROUP BY customer_id
        ) SELECT AVG(total_spend) AS avg_spend FROM customer_spend;
    """
    avg_spend_val = round(pd.read_sql_query(avg_spend_sql, conn).iloc[0, 0], 2)
    print(f"Computed Average Customer Spend: INR {avg_spend_val:,.2f}")
    assert avg_spend_val == 2552.06, f"Average spend mismatch: {avg_spend_val} != 2552.06"

    df_q5 = pd.read_sql_query(queries["Q5"], conn)
    print(f"Customers above average spend: {len(df_q5)}")
    assert len(df_q5) == 550, f"Q5 expected 550 customers, got {len(df_q5)}"
    print(">> Q5 Checkpoints: PASS (550 customers, average spend INR 2,552.06)")

    # -------------------------------------------------------------
    # Q6: Ranking Window Functions (City Leaderboards and Ties)
    # -------------------------------------------------------------
    print("\n[Q6: City Ranking Window Functions (RANK, DENSE_RANK, ROW_NUMBER)]")
    df_q6 = pd.read_sql_query(queries["Q6"], conn)
    
    # Top customer per city
    top_per_city = df_q6[df_q6["spend_rank"] == 1]
    top_map = {row["city"]: (row["customer_id"], round(row["total_spend"], 2)) for _, row in top_per_city.iterrows()}
    
    expected_top = {
        "Bengaluru": ("C0082", 12856.20),
        "Chennai": ("C0930", 10151.40),
        "Delhi": ("C0291", 13742.60),
        "Hyderabad": ("C0743", 12730.05),
        "Mumbai": ("C0693", 14766.55),
        "Pune": ("C0426", 10377.10),
    }
    
    print("Top customer per city:")
    for city, exp in expected_top.items():
        actual = top_map.get(city)
        print(f"  {city:<10}: Customer {actual[0]}, Total Spend INR {actual[1]:,.2f}")
        assert actual == exp, f"Top customer mismatch for {city}: {actual} != {exp}"

    # Difference between RANK() and ROW_NUMBER()
    rank_diff_mask = df_q6["spend_rank"] != df_q6["spend_row_number"]
    rank_diff_count = rank_diff_mask.sum()
    print(f"Rows where RANK() != ROW_NUMBER(): {rank_diff_count}")
    assert rank_diff_count == 8, f"Expected 8 rank-difference rows, got {rank_diff_count}"
    print(">> Q6 Checkpoints: PASS (City leaders verified, exactly 8 tie differences)")

    # -------------------------------------------------------------
    # Q7: Running Total (Daily Revenue and Trajectory)
    # -------------------------------------------------------------
    print("\n[Q7: Daily Revenue & Running Total Window Function]")
    df_q7 = pd.read_sql_query(queries["Q7"], conn)
    print(f"Daily rows: {len(df_q7)}")
    last_cum_rev = round(df_q7.iloc[-1]["cumulative_revenue"], 2)
    print(f"Day 1 ({df_q7.iloc[0]['order_day']}): Revenue = INR {df_q7.iloc[0]['daily_rev']:,.2f}, Cumulative = INR {df_q7.iloc[0]['cumulative_revenue']:,.2f}")
    print(f"Day 183 ({df_q7.iloc[-1]['order_day']}): Revenue = INR {df_q7.iloc[-1]['daily_rev']:,.2f}, Cumulative = INR {last_cum_rev:,.2f}")
    
    assert len(df_q7) == 183, f"Q7 expected 183 daily rows, got {len(df_q7)}"
    assert last_cum_rev == 2998668.80, f"Q7 final cumulative revenue mismatch: {last_cum_rev} != 2998668.80"
    print(">> Q7 Checkpoints: PASS (183 days, final cumulative revenue INR 2,998,668.80)")

    conn.close()

    print("\n" + "=" * 75)
    print("ALL 7 SQL QUERIES EXECUTED AND PASSED ALL REFERENCE CHECKPOINTS!")
    print("=" * 75)


if __name__ == "__main__":
    run_and_validate()

