#!/bin/bash
# Converts .env to env.yaml for Google Cloud Run

INPUT_FILE=".env"
OUTPUT_FILE="env.yaml"

if [ ! -f "$INPUT_FILE" ]; then
  echo ".env file not found!"
  exit 1
fi

echo "# Auto-generated env.yaml from .env" > "$OUTPUT_FILE"
grep -v '^#' "$INPUT_FILE" | grep -v '^$' | while IFS='=' read -r key value; do
  # Remove leading/trailing whitespace from key and value
  key=$(echo "$key" | xargs)
  value=$(echo "$value" | xargs)
  # Quote the value
  echo "$key: \"$value\"" >> "$OUTPUT_FILE"
done

echo "Generated $OUTPUT_FILE"