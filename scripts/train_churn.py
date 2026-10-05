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
import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

load_dotenv()
DB_URL = os.environ["DATABASE_URL"]

CAT = ["country", "acquisition_channel", "plan"]
NUM = ["monthly_price", "tenure_days", "events_30d", "events_prev_30d", "events_change",
       "days_since_last_event", "failed_invoices_90d", "paid_invoices_total",
       "tickets_30d", "tickets_90d"]
FEATURES = CAT + NUM      # churn_type_outcome is deliberately NOT here: it is the answer


def load(name):
    df = pd.read_parquet(f"data/{name}.parquet")
    df[NUM] = df[NUM].apply(pd.to_numeric, errors="coerce")   # Postgres numeric arrives as Decimal
    return df


train, test, score = load("train"), load("test"), load("score")
y_train = train["churned_30d"].astype(int)
y_test = test["churned_30d"].astype(int)
print(f"base churn rate: train={y_train.mean():.3f} test={y_test.mean():.3f}")


def make_pipeline(model, scale):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer(
        [("num", Pipeline(num_steps), NUM),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)]
    )
    return Pipeline([("pre", pre), ("model", model)])


models = {
    "logistic": make_pipeline(LogisticRegression(max_iter=1000), scale=True),
    "gradient_boosting": make_pipeline(
        HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, random_state=42),
        scale=False),
}


def top_decile_lift(y, p):
    """How many times better than random is the riskiest 10% of customers?"""
    k = max(1, len(y) // 10)
    idx = np.argsort(p)[::-1][:k]
    return y.iloc[idx].mean() / y.mean()


results = {}
for name, pipe in models.items():
    pipe.fit(train[FEATURES], y_train)
    p = pipe.predict_proba(test[FEATURES])[:, 1]
    results[name] = (average_precision_score(y_test, p), pipe)
    print(f"{name:18s} ROC-AUC={roc_auc_score(y_test, p):.3f}  "
          f"PR-AUC={average_precision_score(y_test, p):.3f}  "
          f"top-decile lift={top_decile_lift(y_test, p):.1f}x")

best_name = max(results, key=lambda n: results[n][0])
best = results[best_name][1]
print("best model:", best_name)

imp = permutation_importance(best, test[FEATURES], y_test, scoring="average_precision",
                             n_repeats=5, random_state=42)
print(pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).round(4))

# Score today's active customers and write results back to the database
score["churn_prob"] = best.predict_proba(score[FEATURES])[:, 1]
score.to_parquet("data/score_scored.parquet")
joblib.dump(best, "data/churn_model.joblib")

rows = [(int(r.customer_id), r.snapshot_date.date(), round(float(r.churn_prob), 4))
        for r in score.itertuples()]
with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
    cur.executemany(
        """INSERT INTO analytics.churn_scores (customer_id, scored_on, churn_prob)
           VALUES (%s, %s, %s)
           ON CONFLICT (customer_id, scored_on) DO UPDATE SET churn_prob = EXCLUDED.churn_prob""",
        rows)
print(f"wrote {len(rows)} scores")