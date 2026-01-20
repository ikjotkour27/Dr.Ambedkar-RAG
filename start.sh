#!/bin/bash
echo "Starting Dr. Ambedkar RAG API..."
uvicorn api:app --host 0.0.0.0 --port ${PORT}
