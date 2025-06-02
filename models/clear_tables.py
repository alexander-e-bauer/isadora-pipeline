import psycopg2
from config import DATABASE
# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname=DATABASE.DB_NAME,
    user=DATABASE.DB_USER,
    password=DATABASE.DB_PASSWORD,
    host=DATABASE.DB_HOST,
    port="5432"
)
cur = conn.cursor()

# Drop all tables in the `public` schema
drop_tables_query = """
DO
$$
DECLARE
    table_name text;
BEGIN
    -- Loop through all tables
    FOR table_name IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public' -- Change 'public' if using a different schema
    LOOP
        EXECUTE FORMAT('TRUNCATE TABLE %I RESTART IDENTITY CASCADE', table_name);
    END LOOP;
END;
$$;
   """
cur.execute(drop_tables_query)
conn.commit()

print("All tables have been dropped!")
cur.close()
conn.close()
