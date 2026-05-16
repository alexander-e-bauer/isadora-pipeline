#!/bin/bash

# --- CONFIGURE THESE VARIABLES ---
PROJECT_ID="trusty-spanner-446802-p2"
REGION="us-central1"
SERVICE_NAME="isadora-pipeline"
CLOUD_SQL_CONNECTION_NAME="trusty-spanner-446802-p2:us-central1:isadora-v2-db"
IMAGE="us-central1-docker.pkg.dev/$PROJECT_ID/isadora-pipeline-repo/$SERVICE_NAME"


# --- ENSURE .env FILE EXISTS ---
if [ ! -f .env ]; then
  echo ".env file not found! Please create one before deploying."
  exit 1
fi

# --- AUTHENTICATE WITH GCP ---
gcloud config set project $PROJECT_ID

# --- BUILD AND PUSH DOCKER IMAGE ---
gcloud builds submit --tag $IMAGE

# --- READ ENV VARS FROM .env AND DEPLOY TO CLOUD RUN ---
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --add-cloudsql-instances $CLOUD_SQL_CONNECTION_NAME \
  --env-vars-file env.yaml \
  --allow-unauthenticated

# --- PRINT SERVICE URL ---
gcloud run services describe $SERVICE_NAME --region $REGION --format "value(status.url)"
