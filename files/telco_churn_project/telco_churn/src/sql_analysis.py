"""
sql_analysis.py — Load clean data into SQLite and run analytical queries.

Creates:
  data/telco.db                  — SQLite database
  outputs/sql_results/           — one CSV per query result

Queries:
  Q1. Churn rate by contract type
  Q2. Top 10 cities by churn count (if city available)
  Q3. Average revenue by churn status and internet service
  Q4. High-risk customer profile (churn + month-to-month + fiber + e-check)
  Q5. Tenure bucket churn analysis
  Q6. Monthly revenue at risk per payment method

Usage:
    python src/sql_analysis.py
"""

import sqlite3
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLEAN_CSV, DATA_DIR, OUTPUT_DIR
from feature_engineering import add_features

DB_PATH  = DATA_DIR / "telco.db"
SQL_OUT  = OUTPUT_DIR / "sql_results"
SQL_OUT.mkdir(parents=True, exist_ok=True)


def create_database(df: pd.DataFrame) -> sqlite3.Connection:
    """Create (or overwrite) SQLite DB and load customers table."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    print(f"  Database created: {DB_PATH}")
    print(f"  Table 'customers': {len(df):,} rows, {len(df.columns)} columns")
    return conn


QUERIES = {
    "q1_churn_by_contract": """
        SELECT
            Contract,
            COUNT(*)                                          AS total_customers,
            SUM("Churn Value")                               AS churned,
            ROUND(AVG("Churn Value") * 100, 2)               AS churn_rate_pct,
            ROUND(AVG("Monthly Charges"), 2)                 AS avg_monthly_charges
        FROM customers
        GROUP BY Contract
        ORDER BY churn_rate_pct DESC
    """,

    "q2_churn_by_internet_service": """
        SELECT
            "Internet Service",
            COUNT(*)                                          AS total_customers,
            SUM("Churn Value")                               AS churned,
            ROUND(AVG("Churn Value") * 100, 2)               AS churn_rate_pct,
            ROUND(AVG("Monthly Charges"), 2)                 AS avg_monthly_charges,
            ROUND(AVG("Tenure Months"), 1)                   AS avg_tenure_months
        FROM customers
        GROUP BY "Internet Service"
        ORDER BY churn_rate_pct DESC
    """,

    "q3_revenue_by_churn_and_internet": """
        SELECT
            "Churn Value"                                     AS churned,
            "Internet Service",
            COUNT(*)                                          AS customers,
            ROUND(SUM("Monthly Charges"), 2)                 AS total_monthly_rev,
            ROUND(AVG("Monthly Charges"), 2)                 AS avg_monthly_charges,
            ROUND(AVG("Total Charges"), 2)                   AS avg_total_charges
        FROM customers
        GROUP BY "Churn Value", "Internet Service"
        ORDER BY churned DESC, total_monthly_rev DESC
    """,

    "q4_high_risk_profile": """
        SELECT
            Contract,
            "Payment Method",
            "Internet Service",
            COUNT(*)                                          AS customers,
            SUM("Churn Value")                               AS churned,
            ROUND(AVG("Churn Value") * 100, 2)               AS churn_rate_pct,
            ROUND(AVG("Monthly Charges"), 2)                 AS avg_monthly_charges
        FROM customers
        WHERE Contract = 'Month-to-month'
          AND "Payment Method" = 'Electronic check'
        GROUP BY Contract, "Payment Method", "Internet Service"
        ORDER BY churn_rate_pct DESC
    """,

    "q5_tenure_bucket_churn": """
        SELECT
            CASE
                WHEN "Tenure Months" BETWEEN 0  AND 11 THEN '00-11 months'
                WHEN "Tenure Months" BETWEEN 12 AND 23 THEN '12-23 months'
                WHEN "Tenure Months" BETWEEN 24 AND 35 THEN '24-35 months'
                WHEN "Tenure Months" BETWEEN 36 AND 47 THEN '36-47 months'
                WHEN "Tenure Months" BETWEEN 48 AND 59 THEN '48-59 months'
                ELSE '60+ months'
            END                                               AS tenure_bucket,
            COUNT(*)                                          AS total_customers,
            SUM("Churn Value")                               AS churned,
            ROUND(AVG("Churn Value") * 100, 2)               AS churn_rate_pct,
            ROUND(AVG("Monthly Charges"), 2)                 AS avg_monthly_charges
        FROM customers
        GROUP BY tenure_bucket
        ORDER BY tenure_bucket
    """,

    "q6_revenue_at_risk_by_payment": """
        SELECT
            "Payment Method",
            COUNT(*)                                          AS total_customers,
            SUM("Churn Value")                               AS churned,
            ROUND(AVG("Churn Value") * 100, 2)               AS churn_rate_pct,
            ROUND(SUM("Monthly Charges"), 2)                 AS total_monthly_rev,
            ROUND(SUM("Monthly Charges") * AVG("Churn Value"), 2) AS revenue_at_risk
        FROM customers
        GROUP BY "Payment Method"
        ORDER BY revenue_at_risk DESC
    """,

    "q7_senior_citizen_analysis": """
        SELECT
            "Senior Citizen",
            Contract,
            COUNT(*)                                          AS customers,
            ROUND(AVG("Churn Value") * 100, 2)               AS churn_rate_pct,
            ROUND(AVG("Monthly Charges"), 2)                 AS avg_monthly_charges,
            ROUND(AVG("Tenure Months"), 1)                   AS avg_tenure
        FROM customers
        GROUP BY "Senior Citizen", Contract
        ORDER BY "Senior Citizen" DESC, churn_rate_pct DESC
    """,
}


def run_queries(conn: sqlite3.Connection) -> dict:
    results = {}
    for name, sql in QUERIES.items():
        df = pd.read_sql_query(sql, conn)
        results[name] = df
        out = SQL_OUT / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"\n  [{name}]")
        print(df.to_string(index=False))
        print(f"  -> Saved: {out}")
    return results


def run_sql_analysis():
    print("=" * 55)
    print("  SQL ANALYSIS LAYER (SQLite)")
    print("=" * 55)

    df   = pd.read_csv(CLEAN_CSV)
    df   = add_features(df)
    conn = create_database(df)

    print(f"\nRunning {len(QUERIES)} analytical queries...\n")
    results = run_queries(conn)
    conn.close()

    print(f"\n✓ All query results saved to {SQL_OUT}\n")
    return results


if __name__ == "__main__":
    run_sql_analysis()
