import psycopg2
from datetime import datetime, timezone, timedelta
import pytz

from config import DATABASE

# --- USER SETTINGS ---
TABLE_NAME = "historical_data"  # <-- Change this to your table name
TIMESTAMP_COLUMN = "timestamp"  # <-- Change this to your timestamp column


def get_latest_market_30min_timestamp(now=None,
                                      start_hour=6,  # Earliest update hour (inclusive, 24h UTC)
                                      end_hour=23,  # Latest update hour (inclusive, 24h UTC)
                                      ):
    """
    Returns the Unix timestamp for the most recent 30-min mark within the allowed window
    (on a weekday). If outside the window or on a weekend, rolls back to the last valid window.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    dt = now.replace(second=0, microsecond=0)
    minute = (dt.minute // 30) * 30  # Adjust to the nearest 30-minute increment
    dt = dt.replace(minute=minute)

    # Roll back if it's weekend
    while dt.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        dt -= timedelta(minutes=30)
        minute = (dt.minute // 30) * 30
        dt = dt.replace(minute=minute)

    # Roll back if outside update window
    while not (start_hour <= dt.hour <= end_hour):
        dt -= timedelta(minutes=30)
        minute = (dt.minute // 30) * 30
        dt = dt.replace(minute=minute)
        # If we roll into weekend, keep rolling back
        while dt.weekday() >= 5:
            dt -= timedelta(minutes=30)
            minute = (dt.minute // 30) * 30
            dt = dt.replace(minute=minute)

    return int(dt.timestamp())


REFERENCE_TIMESTAMP = get_latest_market_30min_timestamp()  # <-- Your reference timestamp
print(f"Most recent 30-min timestamp: {REFERENCE_TIMESTAMP} ({datetime.utcfromtimestamp(REFERENCE_TIMESTAMP)})")

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname=DATABASE.DB_NAME,
    user=DATABASE.DB_USER,
    password=DATABASE.DB_PASSWORD,
    host=DATABASE.DB_HOST,
    port="5432"
)
cur = conn.cursor()

# 1. Find the most recent timestamp in the table
cur.execute(f"SELECT MAX({TIMESTAMP_COLUMN}) FROM {TABLE_NAME};")
most_recent_ts = cur.fetchone()[0]

if most_recent_ts is None:
    print(f"No data in table '{TABLE_NAME}'.")
else:
    print(f"Most recent timestamp in table: {most_recent_ts} ({datetime.utcfromtimestamp(most_recent_ts)})")
    print(f"Reference timestamp: {REFERENCE_TIMESTAMP} ({datetime.utcfromtimestamp(REFERENCE_TIMESTAMP)})")

    if most_recent_ts == REFERENCE_TIMESTAMP:
        print("✅ Table is up-to-date with the reference timestamp.")
    else:
        print("❌ Table is NOT up-to-date with the reference timestamp.")

cur.close()
conn.close()
