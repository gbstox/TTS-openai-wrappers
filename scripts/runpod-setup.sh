#!/bin/bash
#
# Runpod Setup Helper
#
# This script helps you:
# 1. List existing templates and endpoints
# 2. Create new templates for TTS engines
# 3. Create endpoints using those templates
#
# Usage:
#   ./scripts/runpod-setup.sh list-templates
#   ./scripts/runpod-setup.sh list-endpoints
#   ./scripts/runpod-setup.sh create-template <engine> [image-tag]
#   ./scripts/runpod-setup.sh create-endpoint <engine> <template-id>
#
# Environment:
#   RUNPOD_API_KEY - Required. Your Runpod API key.

set -euo pipefail

REGISTRY="ghcr.io"
OWNER="gbstox"
API_URL="https://rest.runpod.io/v1"

# Check for API key
if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "ERROR: RUNPOD_API_KEY environment variable not set"
    echo "Get your API key from: https://www.runpod.io/console/user/settings"
    exit 1
fi

cmd=${1:-help}

case "$cmd" in
    list-templates)
        echo "Fetching templates..."
        curl -s "${API_URL}/templates" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            | jq '.[] | {id: .id, name: .name, imageName: .imageName}'
        ;;

    list-endpoints)
        echo "Fetching endpoints..."
        curl -s "${API_URL}/endpoints" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            | jq '.[] | {id: .id, name: .name, templateId: .templateId, workersMin: .workersMin, workersMax: .workersMax}'
        ;;

    create-template)
        ENGINE=${2:-}
        TAG=${3:-latest}
        
        if [ -z "$ENGINE" ]; then
            echo "Usage: $0 create-template <engine> [image-tag]"
            echo "Engines: kokoro, cosyvoice, fishspeech, qwen3tts"
            exit 1
        fi
        
        IMAGE="${REGISTRY}/${OWNER}/tts-${ENGINE}:${TAG}"
        TEMPLATE_NAME="TTS-${ENGINE}"
        
        echo "Creating template for ${ENGINE}..."
        echo "Image: ${IMAGE}"
        
        RESPONSE=$(curl -s -X POST "${API_URL}/templates" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            -H "Content-Type: application/json" \
            -d '{
                "name": "'"${TEMPLATE_NAME}"'",
                "imageName": "'"${IMAGE}"'",
                "containerDiskInGb": 30,
                "volumeInGb": 10,
                "volumeMountPath": "/workspace",
                "dockerStartCmd": ["python", "/app/deploy/runpod/handler.py"],
                "isServerless": true,
                "env": {
                    "TTS_ENGINE": "'"${ENGINE}"'",
                    "TTS_PRELOAD_VOICES": "all"
                }
            }')
        
        TEMPLATE_ID=$(echo "$RESPONSE" | jq -r '.id // empty')
        
        if [ -n "$TEMPLATE_ID" ]; then
            echo ""
            echo "SUCCESS! Template created."
            echo ""
            echo "Template ID: ${TEMPLATE_ID}"
            echo ""
            echo "Add this secret to Woodpecker:"
            echo "  Name:  RUNPOD_TEMPLATE_$(echo ${ENGINE} | tr '[:lower:]' '[:upper:]')"
            echo "  Value: ${TEMPLATE_ID}"
        else
            echo "FAILED to create template:"
            echo "$RESPONSE" | jq .
            exit 1
        fi
        ;;

    create-endpoint)
        ENGINE=${2:-}
        TEMPLATE_ID=${3:-}
        
        if [ -z "$ENGINE" ] || [ -z "$TEMPLATE_ID" ]; then
            echo "Usage: $0 create-endpoint <engine> <template-id>"
            exit 1
        fi
        
        ENDPOINT_NAME="tts-${ENGINE}"
        
        echo "Creating endpoint ${ENDPOINT_NAME} with template ${TEMPLATE_ID}..."
        
        RESPONSE=$(curl -s -X POST "${API_URL}/endpoints" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            -H "Content-Type: application/json" \
            -d '{
                "name": "'"${ENDPOINT_NAME}"'",
                "templateId": "'"${TEMPLATE_ID}"'",
                "gpuTypeIds": ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA RTX A4000"],
                "workersMin": 0,
                "workersMax": 3,
                "idleTimeout": 30,
                "flashboot": true,
                "scalerType": "QUEUE_DELAY",
                "scalerValue": 4
            }')
        
        ENDPOINT_ID=$(echo "$RESPONSE" | jq -r '.id // empty')
        
        if [ -n "$ENDPOINT_ID" ]; then
            echo ""
            echo "SUCCESS! Endpoint created."
            echo ""
            echo "Endpoint ID:  ${ENDPOINT_ID}"
            echo "Endpoint URL: https://api.runpod.ai/v2/${ENDPOINT_ID}"
            echo ""
            echo "Test with:"
            echo '  curl -X POST "https://api.runpod.ai/v2/'"${ENDPOINT_ID}"'/runsync" \'
            echo '    -H "Authorization: Bearer $RUNPOD_API_KEY" \'
            echo '    -H "Content-Type: application/json" \'
            echo '    -d '\''{"input": {"input": "Hello world!", "voice": "af_heart"}}'\'''
        else
            echo "FAILED to create endpoint:"
            echo "$RESPONSE" | jq .
            exit 1
        fi
        ;;

    update-template)
        TEMPLATE_ID=${2:-}
        IMAGE=${3:-}
        
        if [ -z "$TEMPLATE_ID" ] || [ -z "$IMAGE" ]; then
            echo "Usage: $0 update-template <template-id> <image-name>"
            exit 1
        fi
        
        echo "Updating template ${TEMPLATE_ID} with image ${IMAGE}..."
        
        RESPONSE=$(curl -s -X PATCH "${API_URL}/templates/${TEMPLATE_ID}" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            -H "Content-Type: application/json" \
            -d '{"imageName": "'"${IMAGE}"'"}')
        
        echo "$RESPONSE" | jq '{id: .id, name: .name, imageName: .imageName}'
        ;;

    help|*)
        echo "Runpod Setup Helper"
        echo ""
        echo "Commands:"
        echo "  list-templates              List all templates"
        echo "  list-endpoints              List all endpoints"
        echo "  create-template <engine>    Create a template for an engine"
        echo "  create-endpoint <engine> <template-id>  Create an endpoint"
        echo "  update-template <id> <image>  Update template image"
        echo ""
        echo "Environment:"
        echo "  RUNPOD_API_KEY - Your Runpod API key (required)"
        ;;
esac
