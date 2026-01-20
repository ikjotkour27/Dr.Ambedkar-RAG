#!/usr/bin/env bash
set -e

echo "Starting Dr. Ambedkar RAG API..."
echo "PORT is: $PORT"

exec uvicorn api:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1
