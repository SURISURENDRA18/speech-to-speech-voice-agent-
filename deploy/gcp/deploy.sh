#!/usr/bin/env bash
# Copy this project to the VM and (re)start the stack.
# Run from your Mac, in the project root:  bash deploy/gcp/deploy.sh
set -euo pipefail

ZONE="${ZONE:-asia-south1-a}"
NAME="${NAME:-voice-agent-gpu}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ARCHIVE="$(mktemp -t voice-agent-XXXX).tgz"

echo "Packing project (without .venv, caches, tests)..."
tar -C "$ROOT" -czf "$ARCHIVE" \
  --exclude .venv --exclude .git --exclude '*.zip' --exclude __pycache__ \
  --exclude .pytest_cache --exclude .ruff_cache --exclude .env \
  --exclude deploy/gcp/svara-tts-inference --exclude deploy/gcp/.env \
  main.py voice_agent deploy .dockerignore

echo "Uploading to $NAME..."
gcloud compute scp --zone "$ZONE" "$ARCHIVE" "$NAME":~/voice-agent.tgz
rm -f "$ARCHIVE"

echo "Running setup on the VM (first run downloads ~25 GB of images/models)..."
gcloud compute ssh "$NAME" --zone "$ZONE" --command \
  "mkdir -p ~/voice-agent && tar -xzf ~/voice-agent.tgz -C ~/voice-agent && bash ~/voice-agent/deploy/gcp/setup_vm.sh"
