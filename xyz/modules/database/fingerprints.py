
from config import DATABASE
import psycopg2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tabulate import tabulate
import json
from threading import Thread
import psycopg2
from psycopg2.extras import Json
scaler = MinMaxScaler()
pca = PCA(n_components=2)  # For dimensionality reduction to 2D
from flask_socketio import SocketIO
import select
# Flask-SocketIO instance will be passed from app.py
socketio = None


def start_listener_thread():
    """
    Start the PostgreSQL listener in a background thread.
    """
    thread = Thread(target=listen_to_postgres)
    thread.daemon = True
    thread.start()
    print("PostgreSQL listener started.")


def initialize_socketio(socketio_instance):
    """
    Initialize the SocketIO instance for use in this module.
    """
    global socketio
    socketio = socketio_instance

def listen_to_postgres():
    """
    Listen for PostgreSQL notifications and broadcast them to WebSocket clients.
    """
    try:
        # Connect to your PostgreSQL database
        conn = psycopg2.connect(
            host=DATABASE.host,
            dbname=DATABASE.name,
            user=DATABASE.user,
            password=DATABASE.password
        )
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Listen to the 'fingerprint_channel'
        cursor.execute("LISTEN fingerprints_channel;")
        print("Listening for notifications on fingerprints_channel...")

        while True:
            if select.select([conn], [], [], 5) == ([], [], []):
                print("Waiting for new events...")
            else:
                conn.poll()  # Check for updates
                while conn.notifies:
                    notification = conn.notifies.pop(0)
                    print("Event received:", notification.payload)


    except Exception as e:
        print(f"Error in PostgreSQL listener: {e}")


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
        cursor.execute("SELECT * FROM fingerprints ORDER BY id ASC")
        rows = cursor.fetchall()

        # Close the connection
        cursor.close()
        conn.close()

        # Return the rows
        return rows
    except Exception as e:
        print(f"Error fetching from database: {e}")
        return []

#start_listener_thread()
# Fetch fingerprints
# fingerprints = fetch_fingerprints()
