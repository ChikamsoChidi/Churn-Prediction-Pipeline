"""
This script exports customer analytics and churn trend data from PostgreSQL to CSV files for Looker reporting.

It connects to the database using credentials loaded from environment variables, executes queries for customer 
details and historical churn trends, formats date columns to YYYYMMDD string representations, and saves the 
results into an exports/ directory as CSV files.
"""

import os
import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.environ["DATABASE_URL"]
os.makedirs("exports", exist_ok=True)

QUERIES = {
    "dash_customers": """
        SELECT customer_id, country, acquisition_channel, plan, monthly_price,
               scored_on, churn_prob, segment_id, tenure_days, risk_band
        FROM analytics.dash_customers""",
    "dash_churn_trend": """
        SELECT snapshot_date, active_customers, churned
        FROM analytics.dash_churn_trend ORDER BY snapshot_date""",
}

with psycopg.connect(DB_URL) as conn:
    for name, sql in QUERIES.items():
        cur = conn.execute(sql)
        df = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
        for col in ("scored_on", "snapshot_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime("%Y%m%d")
        df.to_csv(f"exports/{name}.csv", index=False)
        print(f"{name}: {len(df)} rows")