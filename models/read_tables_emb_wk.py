import psycopg2
import pandas as pd
from config import DATABASE

# --- USER SETTINGS ---
TABLE_NAME = "market_emb_week"  # <-- Change this to your table name

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname=DATABASE.DB_NAME,
    user=DATABASE.DB_USER,
    password=DATABASE.DB_PASSWORD,
    host=DATABASE.DB_HOST,
    port="5432"
)

# Fetch the entire table into a DataFrame
try:
    query = f"SELECT * FROM {TABLE_NAME};"  # Select all columns and rows from the table
    df = pd.read_sql_query(query, conn)  # Use Pandas to read the SQL query result into a DataFrame

    # Ensure the DataFrame displays fully without truncation
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.expand_frame_repr', False)  # Disable wrapping to newline for wide tables

    # Print the DataFrame
    print(df)

finally:
    conn.close()
