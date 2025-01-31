
from config import DATABASE
import psycopg2

def fetch_fingerprints():
    """
    Fetches all bot fingerprints from the PostgreSQL database.
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DATABASE.host,
            dbname=DATABASE.name,
            user=DATABASE.user,
            password=DATABASE.password
        )
        cursor = conn.cursor()

        # Fetch all fingerprints
        cursor.execute("SELECT * FROM bot_fingerprints")
        rows = cursor.fetchall()

        # Close the connection
        cursor.close()
        conn.close()

        # Return the rows
        return rows
    except Exception as e:
        print(f"Error fetching from database: {e}")
        return []

fingerprint = fetch_fingerprints()
print(fingerprint)