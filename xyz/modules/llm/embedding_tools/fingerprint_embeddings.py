


import json
import psycopg2

import tiktoken
import pandas as pd
import re
import ipinfo
from datetime import datetime
import requests
import pprint
from config import DATABASE, OAI, AIS, PINECONE_API_KEY, PINECONE_HOST
from psycopg2.extras import Json, RealDictCursor

from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_HOST)
print("Pinecone.io Index Loaded")



# Dynamically sanitize metadata
def sanitize_metadata(metadata):
    return {k: (v if v is not None else ("" if isinstance(v, str) else 0)) for k, v in metadata.items()}

def save_to_pinecone(fingerprint, embedding, request_id=None, cluster_id=None, reputation_score=None, namespace="default"):
    """
    Saves a bot fingerprint and its embedding to PostgreSQL and Pinecone.

    Parameters:
    - fingerprint (dict): The enriched bot fingerprint JSON object.
    - embedding (list): The embedding array (e.g., vector).
    - request_id (int): ID referencing the associated request (optional).
    - cluster_id (int): ID of the cluster this fingerprint belongs to (optional).
    - reputation_score (float): Reputation score for the fingerprint (optional).
    - namespace (str): Namespace for organizing records in Pinecone (default is "default").
    """
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host=DATABASE.DB_HOST,
            dbname=DATABASE.DB_NAME,
            user=DATABASE.DB_USER,
            password=DATABASE.DB_PASSWORD
        )
        cursor = conn.cursor()

        # Insert the fingerprint into the `fingerprints` table
        try:
            # Insert the fingerprint into the database
            cursor.execute("""
                INSERT INTO fingerprints (request_id, fingerprint, cluster_id, reputation_score)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (
                request_id,
                Json(fingerprint),
                cluster_id,
                reputation_score
            ))

            # Get the generated ID
            fingerprint_id = cursor.fetchone()[0]

            # Sanitize metadata
            metadata = sanitize_metadata({
                "request_id": request_id,
                "cluster_id": cluster_id,
                "reputation_score": reputation_score
            })

            # Format the data for Pinecone
            record = {
                "id": str(fingerprint_id),
                "values": embedding,
                "metadata": metadata
            }

            # Upsert the record into Pinecone
            index.upsert(vectors=[record], namespace=namespace)

            # Commit the transaction
            conn.commit()

            print(f"Fingerprint saved to the database with ID {fingerprint_id}, and embedding saved to Pinecone.")
        except Exception as e:
            conn.rollback()  # Rollback the transaction in case of an error
            print(f"Error: {e}")
        finally:
            cursor.close()
            conn.close()

    except psycopg2.Error as db_error:
        print(f"Database error: {db_error}")
    except Exception as e:
        print(f"Error saving to Pinecone: {e}")


def save_bot_request(fingerprint):
    """
    Saves bot requests into the 'bot_requests' table in the PostgreSQL database.

    Parameters:
    - fingerprint (dict): The enriched bot fingerprint JSON object.
    """
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host=DATABASE.DB_HOST,
            dbname=DATABASE.DB_NAME,
            user=DATABASE.DB_USER,
            password=DATABASE.DB_PASSWORD
        )
        cursor = conn.cursor()

        # Extract required fields from the 'fingerprint' dictionary
        timestamp = fingerprint.get("timestamp")
        ip_address = fingerprint.get("ip")
        user_agent = fingerprint.get("user_agent")
        path = fingerprint.get("path")
        headers = Json(fingerprint.get("headers", {}))  # Serialize headers as JSON
        origin = fingerprint.get("origin")
        is_blacklisted = fingerprint.get("is_blacklisted", False)
        connection_type = fingerprint.get("connection_type", "Unknown")
        tls_analysis = Json(fingerprint.get("tls_analysis", {}))  # Serialize as JSON
        dns_reverse_lookup = fingerprint.get("dns_reverse_lookup", "")
        geo_info = Json(fingerprint.get("geo_info", {}))  # Serialize as JSON
        asn_info = Json(fingerprint.get("asn_info", {}))  # Serialize as JSON
        inconsistent_headers = fingerprint.get("header_inconsistencies", False)
        honeypot_triggered = fingerprint.get("honeypot_interaction", False)
        js_challenge_passed = fingerprint.get("js_challenge_passed", False)
        response_delay = fingerprint.get("response_delay", "00:00:00")

        # Insert the data into the 'bot_requests' table
        cursor.execute("""
            INSERT INTO bot_requests (
                timestamp, ip_address, user_agent, path, headers, origin, 
                is_blacklisted, connection_type, tls_analysis, dns_reverse_lookup, 
                geo_info, asn_info, inconsistent_headers, honeypot_triggered, 
                js_challenge_passed, response_delay
            ) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            timestamp,
            ip_address,
            user_agent,
            path,
            headers,
            origin,
            is_blacklisted,
            connection_type,
            tls_analysis,
            dns_reverse_lookup,
            geo_info,
            asn_info,
            inconsistent_headers,
            honeypot_triggered,
            js_challenge_passed,
            response_delay
        ))

        # Commit the transaction and close the connection
        conn.commit()
        cursor.close()
        conn.close()
        print("Bot request saved to the database.")
    except Exception as e:
        print(f"Error saving bot request to database: {e}")



def get_asn_info(ip):
    """
    Fetch ASN information from the IPinfo API and return it in a neat dictionary format.

    Parameters:
    - ip (str): The Autonomous System Number (e.g., 'ip_address').

    Returns:
    - dict: ASN information in a structured format (keys are 'asn', 'name', 'country', 'registry', 'allocated').
    - None: If the API request fails.
    """
    print(f"fingerprint_embeddings.py line 75\n IP: {ip} \n")
    try:
        # Fetch ASN info from the API

        handler = ipinfo.getHandler(AIS.IPINFO_TOKEN)
        details = handler.getDetails(ip)
        pprint.pprint(details.all)
        return details

    except Exception as e:
        print(f"Error: {e}")
        return None


def is_blacklisted_ip(ip_address):
    """
       Check if the given IP address is blacklisted using a PostgreSQL database.

       Parameters:
       - ip_address (str): The IP address to check.

       Returns:
       - bool: True if the IP is blacklisted, False otherwise.
       """
    try:
        # Connect to the Postgres database
        conn = psycopg2.connect(
            dbname=DATABASE.DB_NAME,
            user=DATABASE.DB_USER,
            password=DATABASE.DB_PASSWORD,
            host=DATABASE.DB_HOST,  # Or the database host
            port="5432"  # Default PostgreSQL port
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Query to check if the IP is blacklisted
        query = "SELECT * FROM ip_blacklist WHERE ip_address = %s"
        cursor.execute(query, (ip_address,))
        result = cursor.fetchone()

        # Close the connection
        cursor.close()
        conn.close()

        # Return True if the IP is found, False otherwise
        return result is not None

    except Exception as e:
        print(f"Error querying the database: {e}")
        return False


def remove_stuff(text: str) -> str:
    """Remove punctuation (except in URLs), newline, tab characters, and large spaces."""
    # Pattern to identify URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Find all URLs using the pattern
    urls = re.findall(url_pattern, text)
    # Replace URLs with a placeholder to avoid altering them
    placeholder = "URL_PLACEHOLDER"
    for url in urls:
        text = text.replace(url, placeholder)

    # Remove large spaces (5 or more spaces)
    text = re.sub(r' {5,}', ' ', text)

    # Restore URLs from placeholders
    for url in urls:
        text = text.replace(placeholder, url, 1)

    return text

def num_tokens(text):
    try:
        # Get the encoding for the model
        encoding = tiktoken.encoding_for_model('gpt-4o')
        # Encode the text to count tokens
        encoded_text = encoding.encode(text, disallowed_special=())
        token_count = len(encoded_text)
        print(f"Token count: {token_count}")
        return token_count
    except Exception as e:
        print(f"Error in num_tokens: {e}")
        return None


def get_embedding(text_to_embed):
    """
    Generates an embedding for the given text using OpenAI's API.
    """
    text_to_embed = remove_stuff(text_to_embed)
    # Check the number of tokens
    token_count = num_tokens(text_to_embed)
    max_token_limit = 8192  # Adjust based on your model's token limit

    if token_count is None or token_count > max_token_limit:
        print(f"Text exceeds the token limit ({max_token_limit} tokens). Skipping embedding.")
        return None

    try:
        # Embed a line of text
        response = OAI.client.embeddings.create(
            model=OAI.embedding3,
            input=[text_to_embed]
        )
        # Extract the AI output embedding as a list of floats
        embedding = response.data[0].embedding
        print(f"---\nEmbedding generated successfully for text: {text_to_embed[:100]}...")
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def calculate_request_frequency(ip_address):
    # Query your database to count requests from the same IP in the last minute
    conn = psycopg2.connect(
        dbname=DATABASE.DB_NAME,
        user=DATABASE.DB_USER,
        password=DATABASE.DB_PASSWORD,
        host=DATABASE.DB_HOST
    )
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) 
        FROM bot_requests 
        WHERE ip_address = %s AND timestamp >= NOW() - INTERVAL '1 minute'
    """, (ip_address,))
    frequency = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return frequency



def preprocess_fingerprint(bot_fingerprint):
    """
    Preprocesses the bot fingerprint data.

    Parameters:
        bot_fingerprint (dict): The raw bot fingerprint provided by `fingerprint_bot`.

    Returns:
        dict: The preprocessed fingerprint data, ready for further analysis.
    """
    try:
        # Validate the input
        if not isinstance(bot_fingerprint, dict):
            raise ValueError("Expected a dictionary for bot fingerprint data")

        # Extract required fields
        ip = bot_fingerprint.get("ip")
        geo_info = bot_fingerprint.get("geo_info", {})
        asn_info = bot_fingerprint.get("asn_info", {})
        headers = bot_fingerprint.get("headers", {})
        user_agent = bot_fingerprint.get("user_agent", "")
        timestamp = bot_fingerprint.get("timestamp")
        request_frequency = bot_fingerprint.get("request_frequency", 0)
        is_blacklisted = bot_fingerprint.get("is_blacklisted", False)
        connection_type = bot_fingerprint.get("connection_type", "Unknown")
        tls_analysis = bot_fingerprint.get("tls_analysis", {})
        dns_reverse_lookup = bot_fingerprint.get("dns_reverse_lookup", "")
        header_inconsistencies = bot_fingerprint.get("header_inconsistencies", False)
        honeypot_interaction = bot_fingerprint.get("honeypot_interaction", False)
        js_challenge_passed = bot_fingerprint.get("js_challenge_passed", False)
        response_delay = bot_fingerprint.get("response_delay", "00:00:00")

        # Validate the IP address
        print(f"Break: {ip}")
        if not ip or not isinstance(ip, str):
            raise ValueError("Invalid or missing IP address in bot fingerprint")

        # Verify if the IP is blacklisted
        if is_blacklisted_ip(ip):
            raise ValueError(f"IP address {ip} is blacklisted")

        # Preprocess request headers (e.g., normalize header keys)
        normalized_headers = {key.lower(): value for key, value in headers.items()}

        # Clean up geo_info and asn_info (if necessary)
        geo_info_cleaned = {k: v for k, v in geo_info.items() if v}  # Remove empty values
        asn_info_cleaned = {k: v for k, v in asn_info.items() if v}  # Remove empty values

        # Process TLS Analysis (if present, keep only relevant fields)
        tls_cleaned = {
            "protocol": tls_analysis.get("protocol", "Unknown"),
            "cipher_suite": tls_analysis.get("cipher_suite", "Unknown")
        }

        # Convert response_delay into a duration in seconds (if necessary)
        if isinstance(response_delay, str):
            try:
                h, m, s = map(int, response_delay.split(":"))
                response_delay_seconds = h * 3600 + m * 60 + s
            except ValueError:
                response_delay_seconds = 0
        else:
            response_delay_seconds = 0

        # Build the final preprocessed fingerprint
        preprocessed_fingerprint = {
            "ip": ip,
            "geo_info": geo_info_cleaned,
            "asn_info": asn_info_cleaned,
            "headers": normalized_headers,
            "user_agent": user_agent,
            "timestamp": timestamp,
            "request_frequency": request_frequency,
            "is_blacklisted": is_blacklisted,
            "connection_type": connection_type,
            "tls_analysis": tls_cleaned,
            "dns_reverse_lookup": dns_reverse_lookup,
            "header_inconsistencies": header_inconsistencies,
            "honeypot_interaction": honeypot_interaction,
            "js_challenge_passed": js_challenge_passed,
            "response_delay_seconds": response_delay_seconds,
        }

        return preprocessed_fingerprint

    except Exception as e:
        # Log the error and return None
        print(f"Error in preprocess_fingerprint: {str(e)}")
        return None



def generate_fingerprint_embedding(fingerprint):
    """
    Generates an embedding for a bot fingerprint.
    """
    # Preprocess fingerprint into a normalized string
    fingerprint_json = preprocess_fingerprint(fingerprint)
    fingerprint_str = json.dumps(fingerprint_json)
    # Generate embedding
    embedding = get_embedding(fingerprint_str)
    return embedding


def fingerprint_bot(request, origin, user_agent):
    if not origin or not isinstance(origin, str):
        raise ValueError("Invalid or missing IP address.")

    if origin == "127.0.0.1":
        origin = "73.18.165.209"

    details = get_asn_info(origin)
    if not details:
        print(f"Could not retrieve ASN info for IP: {origin}")

    print(f"City: {details.city}")

    geo_info = {
        "city": details.city if details and details.city else "Unknown",
        "region": details.region if details and details.region else "Unknown",
        "country": details.country_name if details and details.country_name else "Unknown",
        "loc": details.loc if details and details.loc else "0,0",
        "timezone": details.timezone if details and details.timezone else "UTC",
        "postal": details.postal if details and details.postal else "00000",
    }
    asn_info = {
        "hostname": details.hostname if details and details.hostname else "Unknown",
        "org": details.org if details and details.org else "Unknown",
    }

    is_blacklisted = is_blacklisted_ip(origin)

    # Enriched fingerprint
    bot_fingerprint = {
        "ip": origin,
        "geo_info": geo_info,
        "asn_info": asn_info,
        "is_blacklisted": is_blacklisted,
        "user_agent": user_agent,
        "path": request.path,
        "headers": {k: v for k, v in request.headers.items()},
        "timestamp": datetime.utcnow().isoformat(),
        "connection_type": "VPN",  # Replace with actual detection logic
        "tls_analysis": {
            "protocol": "TLS 1.3",
            "cipher_suite": "TLS_AES_256_GCM_SHA384"
        },
        "dns_reverse_lookup": "bot.example.com",  # Replace with actual DNS lookup
        "header_inconsistencies": True,  # Example logic
        "honeypot_interaction": False,  # Replace with actual honeypot logic
        "js_challenge_passed": False,  # Replace with actual JS challenge result
        "request_frequency": 10,  # Number of requests in the last minute (example)
        "response_delay": "00:00:05"  # Response delay (example)
    }

    # Generate and save embedding
    embedding = generate_fingerprint_embedding(bot_fingerprint)
    save_to_pinecone(bot_fingerprint, embedding, request_id=None, cluster_id=None, reputation_score=None, namespace="default")
    save_bot_request(bot_fingerprint)



#get_asn_info(
 #   ip='34.16.120.105')
