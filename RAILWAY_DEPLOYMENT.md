# Stable Audio Tools Railway Deployment Guide

## Overview
This guide explains how to deploy the Stable Audio Tools microservice to Railway with proper configuration for model loading and dependency management.

## Issues Fixed

### 1. Flash Attention Dependency
- **Problem**: `No module named 'flash_attn'` error during startup
- **Solution**: 
  - Added flash-attn as optional dependency in nixpacks.toml
  - Installation fails gracefully if flash-attn cannot be compiled
  - Service continues to work without flash-attn (with reduced performance)

### 2. PyTorch Version Compatibility
- **Problem**: `torch.nn.attention.flex_attention not available in this PyTorch version`
- **Solution**: 
  - Updated PyTorch from 2.1.0 to >=2.2.0 for better compatibility
  - Added proper error handling for missing flex_attention

### 3. Railway Configuration
- **Problem**: Using incorrect Narratix2.0 configuration
- **Solution**: 
  - Created stable-audio-tools specific railway.toml
  - Added proper environment variables for HuggingFace token
  - Configured health check timeout for model loading

### 4. Model Loading
- **Problem**: Model loading failures causing 503 errors
- **Solution**: 
  - Added proper HuggingFace token validation
  - Improved error handling and logging
  - Service starts even if model loading fails (with degraded functionality)

## Required Environment Variables

### Development Environment
```bash
HUGGING_FACE_HUB_TOKEN=hf_your_actual_token_here
SERVICE_TOKEN=dev-stable-audio-token-2024
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### Production Environment
```bash
HUGGING_FACE_HUB_TOKEN=hf_your_actual_token_here
SERVICE_TOKEN=prod-stable-audio-token-2024
ENVIRONMENT=production
LOG_LEVEL=INFO
```

## Deployment Steps

1. **Set up Railway project**:
   - Create new Railway project
   - Connect to stable-audio-tools repository
   - Set environment variables (especially HUGGING_FACE_HUB_TOKEN)

2. **Deploy**:
   - Railway will automatically detect nixpacks.toml
   - Build process will install dependencies with proper timeouts
   - Service will start with startup script

3. **Verify deployment**:
   - Check `/health` endpoint (always returns 200)
   - Check response body for `"model_loaded": true/false`
   - Check response body for `"status": "healthy"/"degraded"`

## Health Check Endpoints

- `GET /health` - Returns detailed service status and model loading state (always 200 status)
- `GET /ping` - Simple connectivity check (always returns 200)
- `GET /` - Basic service info
- `POST /generate` - Audio generation endpoint (requires authentication)

**Important**: The `/health` endpoint now always returns HTTP 200 for Railway compatibility. Service status is indicated in the response body with fields like `status`, `ready`, and `model_loaded`.

## Troubleshooting

### Model Loading Issues
- Ensure HUGGING_FACE_HUB_TOKEN is set correctly
- Check Railway logs for model loading errors
- Verify token has access to stabilityai/stable-audio-open-1.0

### Flash Attention Issues
- Flash attention is optional - service works without it
- If installation fails, check Railway build logs
- Performance may be reduced without flash attention

### Memory Issues
- Railway provides limited memory for free tier
- Consider upgrading to paid plan for better performance
- Monitor memory usage in Railway dashboard

## Performance Notes

- Model loading takes 30-60 seconds on Railway
- First generation request may be slower due to warmup
- Flash attention significantly improves performance when available
- CPU-only inference is slower but functional

## Security

- Service uses bearer token authentication
- Set SERVICE_TOKEN environment variable
- Requests to /generate require: `Authorization: Bearer <token>`
