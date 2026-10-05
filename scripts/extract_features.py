"""
This program extracts the data from the database to be parsed and modelled in Python. 
It pulls customer churn data from a PostgreSQL database, prepares time-based datasets 
for machine learning, and saves them locally as Parquet files which retain the datetime types and 
are faster to read/write.
"""


import os
import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.environ["DATABASE_URL"]

with psycopg.connect(DB_URL) as conn:
    cur = conn.execute("SELECT * FROM analytics.customer_snapshots")
    df = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])

df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

# Time-based split; train on the past, test on the more recent future
labeled = df[df["churned_30d"].notna()]
train = labeled[labeled["snapshot_date"] <  "2026-06-01"]
test  = labeled[labeled["snapshot_date"] >= "2026-06-01"]
score = df[df["snapshot_date"] == df["snapshot_date"].max()]   # today's active customers

os.makedirs("data", exist_ok=True)
train.to_parquet("data/train.parquet")
test.to_parquet("data/test.parquet")
score.to_parquet("data/score.parquet")
print(len(train), len(test), len(score))