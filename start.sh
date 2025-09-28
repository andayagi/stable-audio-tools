#!/bin/bash

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONMALLOC=malloc

# Create cache directories
mkdir -p /tmp/transformers_cache
mkdir -p /tmp/torch_cache

# Start the application
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
