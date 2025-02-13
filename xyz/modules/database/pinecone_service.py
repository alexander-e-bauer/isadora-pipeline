from pinecone import Pinecone
from config import DATABASE
import psycopg2
import pandas as pd
from psycopg2.extensions import connection as Connection
from typing import List, Dict


# Initialize constants
PINECONE_INDEX_NAME = DATABASE.pinecone_host

# Initialize Pinecone
pc = Pinecone(
    api_key=DATABASE.pinecone_api_key
)
index = pc.Index(PINECONE_INDEX_NAME)
print("Pinecone.io Index Loaded")


def get_postgres_connection() -> Connection:
    """
    Establishes and returns a connection to the PostgreSQL database.
    """
    try:
        return psycopg2.connect(
            dbname=DATABASE.name,
            user=DATABASE.user,
            password=DATABASE.password,
            host=DATABASE.host,
        )
    except psycopg2.Error as e:
        raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")


def fetch_fingerprint_ids() -> List[str]:
    """
    Fetches fingerprint IDs from the PostgreSQL database.

    Returns:
        A list of fingerprint IDs.
    """
    query = """
    SELECT id AS fingerprint_id
    FROM fingerprints
    """
    try:
        with get_postgres_connection() as conn:
            # Fetch the IDs as a DataFrame
            df = pd.read_sql(query, conn)
            # Convert the IDs to a list of strings
            return df['fingerprint_id'].astype(str).tolist()
    except Exception as e:
        raise RuntimeError(f"Error fetching fingerprint IDs from PostgreSQL: {e}")


def fetch_fingerprints() -> pd.DataFrame:
    """
    Fetches fingerprint IDs from the PostgreSQL database.

    Returns:
        A list of fingerprint IDs.
    """
    query = """
    SELECT *
    FROM fingerprints
    """
    try:
        with get_postgres_connection() as conn:
            # Fetch the IDs as a DataFrame
            df = pd.read_sql(query, conn)
            # Convert the IDs to a list of strings
            return df
    except Exception as e:
        raise RuntimeError(f"Error fetching fingerprint IDs from PostgreSQL: {e}")


def fetch_embeddings_from_pinecone(fingerprint_ids: List[str]) -> Dict[str, List[float]]:
    """
    Fetches embeddings from Pinecone using a list of fingerprint IDs.

    Args:
        fingerprint_ids: List of fingerprint IDs to fetch embeddings for.

    Returns:
        Dictionary mapping fingerprint IDs to their embeddings.
    """
    try:
        # Fetch data from Pinecone
        result = index.fetch(ids=fingerprint_ids, namespace='default')

        # Ensure result.vectors exists and is not None
        if not hasattr(result, 'vectors') or result.vectors is None:
            raise ValueError("`result.vectors` is missing or None.")

        # Access the 'vectors' attribute of the FetchResponse object
        vectors = result.vectors  # Should be a dictionary-like object

        # Extract embeddings from the vectors
        embeddings = {
            fp_id: vectors[fp_id].values
            for fp_id in fingerprint_ids if fp_id in vectors
        }

        # Log missing IDs
        missing_ids = set(fingerprint_ids) - set(embeddings.keys())
        if missing_ids:
            print(f"Warning: The following fingerprint IDs were not found in Pinecone: {missing_ids}")

        return embeddings
    except Exception as e:
        raise RuntimeError(f"Error fetching embeddings from Pinecone: {e}")


def main() -> pd.DataFrame:
    """
    Main process to fetch fingerprint IDs from PostgreSQL, retrieve embeddings from Pinecone,
    and combine them into a single DataFrame.

    Returns:
        A DataFrame with fingerprint IDs and their embeddings.
    """
    # Fetch all fingerprints from PostgreSQL
    fingerprints = fetch_fingerprints()

    # Fetch fingerprint IDs from the DataFrame
    fingerprint_ids = fingerprints['id'].astype(str).tolist()
    print(f"Fetched {len(fingerprint_ids)} fingerprint IDs from PostgreSQL.")

    # Fetch corresponding embeddings from Pinecone
    pinecone_embeddings = fetch_embeddings_from_pinecone(fingerprint_ids)
    print(f"Fetched {len(pinecone_embeddings)} embeddings from Pinecone.")

    # Ensure the embeddings are aligned with the fingerprints DataFrame
    fingerprints['embedding'] = fingerprints['id'].astype(str).map(pinecone_embeddings)

    # Check for any missing embeddings
    missing_embeddings = fingerprints[fingerprints['embedding'].isnull()]
    if not missing_embeddings.empty:
        print(f"Warning: Missing embeddings for the following IDs:\n{missing_embeddings['id'].tolist()}")

    return fingerprints



if __name__ == "__main__":
    try:
        # Run main process and display the first few rows of the combined data
        final_data = main()
        print(final_data.head())
    except Exception as error:
        print(f"An error occurred: {error}")
