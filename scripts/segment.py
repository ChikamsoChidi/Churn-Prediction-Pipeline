"""
This script performs customer segmentation using K-Means clustering and updates the database.

It reads scored customer data, cleans key metrics, fills missing activity values, and 
engineers a lifetime value feature. It transforms four behavioral features (recency, 
30-day activity, LTV, tenure) using log normalization and standard scaling to handle skew. 

The script tests cluster counts from k=3 to k=8 using silhouette scores to pick the 
optimal k, assigns a segment_id to each customer, prints aggregate profile metrics, and 
bulk-updates the segment_id column in the analytics.churn_scores PostgreSQL table.
"""

import os
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

load_dotenv()
DB_URL = os.environ["DATABASE_URL"]

df = pd.read_parquet("data/score_scored.parquet")
for c in ["monthly_price", "paid_invoices_total", "days_since_last_event"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["days_since_last_event"] = df["days_since_last_event"].fillna(60)   # none seen in the 60-day window
df["lifetime_value"] = df["paid_invoices_total"] * df["monthly_price"]  # approximation: ignores plan changes

cols = ["days_since_last_event", "events_30d", "lifetime_value", "tenure_days"]
X = StandardScaler().fit_transform(np.log1p(df[cols]))   # log tames skew, scaling equalises units

scores = {}
for k in range(3, 9):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    scores[k] = silhouette_score(X, labels)
    print(f"k={k}  silhouette={scores[k]:.3f}")

K = max(scores, key=scores.get)     # override with 4 or 5 if that gives more useful business groups
df["segment_id"] = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X)

profile = df.groupby("segment_id").agg(
    customers=("customer_id", "count"),
    days_since_event=("days_since_last_event", "mean"),
    events_30d=("events_30d", "mean"),
    lifetime_value=("lifetime_value", "mean"),
    tenure_days=("tenure_days", "mean"),
    avg_churn_prob=("churn_prob", "mean"),
).round(2)
print(profile)

rows = [(int(s), int(c), d.date()) for s, c, d in zip(df["segment_id"], df["customer_id"], df["snapshot_date"])]
with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
    cur.executemany(
        "UPDATE analytics.churn_scores SET segment_id = %s WHERE customer_id = %s AND scored_on = %s", rows)