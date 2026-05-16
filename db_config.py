import psycopg2
import os
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

DB_PARAMS = {
    "host": DB_HOST,
    "database": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "port": DB_PORT,
    "sslmode": "require"
}

def init_db():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS inactivity_log (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                zone_id TEXT,
                worker_id TEXT,
                idle_duration_seconds INTEGER,
                image_path TEXT
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print(f"Database '{DB_NAME}' ready (using Image Paths).")
    except Exception as e:
        print(f"Database Init Error: {e}")

def log_idle_event(zone, worker_id, duration, image_path):
    """Saves the idle event details and the local file path to Postgres."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        query = """
            INSERT INTO inactivity_log (zone_id, worker_id, idle_duration_seconds, image_path)
            VALUES (%s, %s, %s, %s)
        """
        cur.execute(query, (zone, f"Worker {worker_id}", int(duration), image_path))
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"Path Logged: {image_path}")
    except Exception as e:
        print(f"Logging Error: {e}")
