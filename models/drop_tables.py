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
   DO $$ DECLARE
       stmt TEXT;
   BEGIN
       FOR stmt IN
           SELECT 'DROP TABLE IF EXISTS "' || tablename || '" CASCADE;'
           FROM pg_tables
           WHERE schemaname = 'public'
       LOOP
           EXECUTE stmt;
       END LOOP;
   END $$;
   """
cur.execute(drop_tables_query)
conn.commit()

print("All tables have been dropped!")
cur.close()
conn.close()
