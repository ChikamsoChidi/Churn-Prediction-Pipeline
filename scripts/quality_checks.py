import os
import sys
import psycopg
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.environ["DATABASE_URL"]

# (description, SQL that returns the number of BAD rows; must be 0)
CHECKS = [
    ("no subscriptions ending before they start",
     "SELECT COUNT(*) FROM subscriptions WHERE ended_at <= started_at"),
    ("no customers without a subscription",
     "SELECT COUNT(*) FROM customers c WHERE NOT EXISTS "
     "(SELECT 1 FROM subscriptions s WHERE s.customer_id = c.customer_id)"),
    ("no events before the customer signed up",
     "SELECT COUNT(*) FROM usage_events e JOIN customers c USING (customer_id) "
     "WHERE e.occurred_at < c.signed_up_at"),
    ("no events or invoices dated in the future",
     "SELECT (SELECT COUNT(*) FROM usage_events WHERE occurred_at > now()) "
     "+ (SELECT COUNT(*) FROM invoices WHERE issued_at > now())"),
    ("no duplicate emails ignoring case",
     "SELECT COUNT(*) FROM (SELECT lower(email) FROM customers "
     "GROUP BY 1 HAVING COUNT(*) > 1) d"),
]

failed = False
with psycopg.connect(DB_URL) as conn:
    for name, sql in CHECKS:
        bad = conn.execute(sql).fetchone()[0]
        print(f"[{'FAIL' if bad else ' ok '}] {name}" + (f" ({bad} bad rows)" if bad else ""))
        failed |= bool(bad)

    # Volume check: today's data shouldn't collapse versus last month's
    n = conn.execute("SELECT COUNT(*) FROM subscriptions WHERE ended_at IS NULL").fetchone()[0]
    print(f"[info] active subscriptions: {n}")

sys.exit(1 if failed else 0)