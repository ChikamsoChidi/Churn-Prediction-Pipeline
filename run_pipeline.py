"""
This script orchestrates and executes the full machine learning churn pipeline end-to-end.

It sequentially runs quality checks, refreshes the analytics.customer_snapshots 
materialized view in PostgreSQL, extracts features, trains and scores the churn model, 
segments active customers, and exports dashboard reports for Looker. The script 
measures execution time for each step and halts pipeline execution immediately if 
any subprocess fails.
"""

import subprocess
import sys
import time
from datetime import datetime

STEPS = [
    ("quality checks",      [sys.executable, "scripts/quality_checks.py"]),
    ("refresh features",    ["psql", "-U", "churn_admin", "-d", "churn", "-h", "localhost",
                             "-c", "REFRESH MATERIALIZED VIEW analytics.customer_snapshots;"]),
    ("extract",             [sys.executable, "scripts/extract_features.py"]),
    ("train + score",       [sys.executable, "scripts/train_churn.py"]),
    ("segment",             [sys.executable, "scripts/segment.py"]),
    ("export",              [sys.executable, "scripts/export_for_looker.py"]),
]

print(f"=== pipeline start {datetime.now():%Y-%m-%d %H:%M} ===")
for name, cmd in STEPS:
    t0 = time.time()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!!! step '{name}' failed (exit {result.returncode}); pipeline stopped")
        sys.exit(result.returncode)
    print(f"--- {name} done in {time.time() - t0:.1f}s")
print("=== pipeline complete ===")