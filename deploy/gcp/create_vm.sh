#!/usr/bin/env bash
# Create the GPU VM (1x NVIDIA L4) in Mumbai with a static IP and HTTP/HTTPS open.
# Run from your Mac:  bash deploy/gcp/create_vm.sh
# Options (env vars): ZONE=asia-south1-a  NAME=voice-agent-gpu  SPOT=true
set -euo pipefail

PROJECT="$(gcloud config get-value project 2>/dev/null)"
ZONE="${ZONE:-asia-south1-a}"
REGION="${ZONE%-*}"   # asia-south1-a -> asia-south1
NAME="${NAME:-voice-agent-gpu}"
SPOT="${SPOT:-false}"
IP_NAME="${NAME}-ip"

echo "Project: $PROJECT  Zone: $ZONE  VM: $NAME  Spot: $SPOT"

# GPUs need project-wide quota > 0 (new and free-trial projects start at 0).
GPU_QUOTA="$(gcloud compute project-info describe --format=json \
  | python3 -c "import json,sys; q=[x for x in json.load(sys.stdin)['quotas'] if x['metric']=='GPUS_ALL_REGIONS']; print(int(q[0]['limit']) if q else 0)")"
if [ "$GPU_QUOTA" -lt 1 ]; then
  cat <<EOF
ERROR: GPUS_ALL_REGIONS quota is $GPU_QUOTA. Request 1 GPU first:
  https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT
  (filter "GPUs (all regions)", edit, set 1). Free-trial accounts must be
  upgraded to a paid account first; remaining credits are kept.
EOF
  exit 1
fi

if ! gcloud compute addresses describe "$IP_NAME" --region "$REGION" >/dev/null 2>&1; then
  gcloud compute addresses create "$IP_NAME" --region "$REGION"
fi
IP="$(gcloud compute addresses describe "$IP_NAME" --region "$REGION" --format='value(address)')"

if ! gcloud compute firewall-rules describe allow-voice-agent-web >/dev/null 2>&1; then
  gcloud compute firewall-rules create allow-voice-agent-web \
    --allow tcp:80,tcp:443 --target-tags voice-agent \
    --description "HTTPS for the voice agent (Caddy)"
fi

SPOT_FLAGS=()
if [ "$SPOT" = "true" ]; then
  # ~60-70% cheaper; Google may stop the VM at any time.
  SPOT_FLAGS=(--provisioning-model SPOT --instance-termination-action STOP)
fi

# g2-standard-8 = 8 vCPU, 32 GB RAM, 1x L4 24 GB (attached automatically).
gcloud compute instances create "$NAME" \
  --zone "$ZONE" \
  --machine-type g2-standard-8 \
  --maintenance-policy TERMINATE \
  --image-family common-cu129-ubuntu-2204-nvidia-580 \
  --image-project deeplearning-platform-release \
  --boot-disk-size 150GB \
  --boot-disk-type pd-balanced \
  --metadata install-nvidia-driver=True \
  --tags voice-agent \
  --address "$IP" \
  ${SPOT_FLAGS[@]+"${SPOT_FLAGS[@]}"}

cat <<EOF

VM created. Public IP: $IP
Your HTTPS URL will be: https://${IP//./-}.sslip.io
Next: bash deploy/gcp/deploy.sh
EOF
