import pandas as pd
import numpy as np
from keras.src.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from keras.api.models import Sequential
from keras.api.optimizers import Adam
from xyz.modules.database.pinecone_service import main as fetch_postgres_data
import json


def generate_anomaly_labels(embedding_matrix):
    """
    Generate binary anomaly labels based on statistical outliers in the embedding matrix.

    Args:
        embedding_matrix (np.ndarray): The embedding matrix.

    Returns:
        np.ndarray: Binary anomaly labels (1 for anomalous, 0 for normal).
    """
    # Compute the mean and standard deviation for each feature
    mean = np.mean(embedding_matrix, axis=0)
    std_dev = np.std(embedding_matrix, axis=0)

    # Identify outliers (e.g., values > 3 standard deviations from the mean)
    z_scores = np.abs((embedding_matrix - mean) / std_dev)
    anomaly_threshold = 3  # Threshold for anomalies
    anomalies = np.any(z_scores > anomaly_threshold, axis=1)

    # Convert to binary labels: 1 = anomalous, 0 = normal
    labels = anomalies.astype(int)
    return labels


def preprocess_dataframe(df):
    """
    Preprocess the input DataFrame for anomaly detection while preserving WIP columns.

    Args:
        df (pd.DataFrame): Input dataframe containing website request data.

    Returns:
        tuple: Preprocessed feature matrix (X_train, X_test) and labels (Y_train, Y_test).
    """
    for column in df.columns:
        print(f"Column: {column}")
        print(df[column].sample(3))  # Display 3 random rows for this column
        print("-" * 40)

    try:
        # Step 1: Process the 'embedding' column (already contains lists)
        if 'embedding' not in df.columns:
            raise ValueError("The 'embedding' column is missing from the DataFrame.")

        embedding_matrix = np.array(df['embedding'].tolist())
        scaler = StandardScaler()
        embedding_matrix = scaler.fit_transform(embedding_matrix)

        # Step 2: Extract Geo info (e.g., 'city') from the 'fingerprint' column
        # Assuming 'fingerprint' contains JSON-like data
        """if 'fingerprint' in df.columns:
            df['city'] = df['fingerprint'].apply(
                lambda x: json.loads(x).get('geo_info', {}).get('city', None) if pd.notnull(x) else None
            )
        else:
            print("Warning: 'fingerprint' column is missing. Skipping geo info extraction.")
            df['city'] = None"""


        X = embedding_matrix

        # Step 2: Generate binary anomaly labels (Y)
        Y = generate_anomaly_labels(embedding_matrix)

        # Placeholder for WIP columns
        # These columns are kept in the DataFrame but not used in the current feature matrix
        if 'request_id' in df.columns:
            print("Note: 'request_id' column is WIP and not included in the feature matrix.")
        if 'cluster_id' in df.columns:
            print("Note: 'cluster_id' column is WIP and not included in the feature matrix.")
        if 'reputation_score' in df.columns:
            print("Note: 'reputation_score' column is WIP and not included in the feature matrix.")

        # Step 3: Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        return X_train, X_test, Y_train, Y_test

    except Exception as e:
        raise ValueError(f"An error occurred during preprocessing: {e}")


def build_model(input_dim):
    """
    Builds a simple neural network for anomaly detection.

    Args:
        input_dim (int): Number of input features.

    Returns:
        model: A compiled Keras model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification (anomalous or normal)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate_model(X_train, X_test, Y_train, Y_test):
    """
    Trains the anomaly detection model and evaluates its performance.

    Args:
        X_train (np.ndarray): Training feature matrix.
        X_test (np.ndarray): Test feature matrix.
        Y_train (np.ndarray): Training labels.
        Y_test (np.ndarray): Test labels.

    Returns:
        model: The trained Keras model.
    """
    # Build the model
    input_dim = X_train.shape[1]
    model = build_model(input_dim)

    # Train the model
    print("\nTraining the model...")
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=32, verbose=1)

    # Evaluate the model
    print("\nEvaluating the model...")
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Make predictions
    predictions = (model.predict(X_test) > 0.5).astype(int).flatten()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(Y_test, predictions, target_names=['Normal', 'Anomalous']))

    # Accuracy score
    acc = accuracy_score(Y_test, predictions)
    print(f"Accuracy Score: {acc:.4f}")

    return model


def main():
    """
    Main function to fetch data, preprocess it, train the model, and evaluate its performance.

    Returns:
        None
    """
    try:
        # Fetch data from Pinecone service
        final_data = fetch_postgres_data()  # pinecone_service.py main() Function

        # Display initial data information
        print(f"Loading Data Type: {type(final_data)}")
        print("\nCombined Data:")
        print(final_data.info())
        print(final_data.shape)
        print(final_data.head())

        # Preprocess the DataFrame
        X_train, X_test, Y_train, Y_test = preprocess_dataframe(final_data)

        # Display shapes of the preprocessed data
        print("\nPreprocessed Data:")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"Y_train shape: {Y_train.shape}, Y_test shape: {Y_test.shape}")

        # Train and evaluate the model
        model = train_and_evaluate_model(X_train, X_test, Y_train, Y_test)

        print("\nModel training and evaluation complete!")

    except Exception as error:
        print(f"An error occurred in anomaly_detector.py: {error}")


# Run the main function
if __name__ == "__main__":
    main()
