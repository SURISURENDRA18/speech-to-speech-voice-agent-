#!/usr/bin/env bash
# Runs ON the VM (called by deploy.sh). Installs Docker + NVIDIA container
# toolkit if missing, fetches the Svara server, and starts the stack.
set -euo pipefail

cd "$(dirname "$0")"

echo "== GPU"
for _ in $(seq 1 30); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if ! command -v docker >/dev/null 2>&1; then
  echo "== Installing Docker"
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER" || true

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "== Installing NVIDIA container toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -y && sudo apt-get install -y nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker >/dev/null
sudo systemctl restart docker

if [ ! -d svara-tts-inference ]; then
  echo "== Fetching Svara inference server"
  git clone --depth 1 https://github.com/Kenpath/svara-tts-inference.git
fi

if [ ! -f .env ]; then
  IP="$(curl -fsS -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip)"
  echo "SITE_ADDRESS=${IP//./-}.sslip.io" > .env
  echo "Wrote .env with SITE_ADDRESS=${IP//./-}.sslip.io (edit it to use your own domain)"
fi

echo "== Building and starting (first start: model downloads take 5-15 min)"
sudo docker compose up -d --build

echo
echo "Watch progress:  sudo docker compose -f ~/voice-agent/deploy/gcp/docker-compose.yml logs -f"
echo "Open:            https://$(grep SITE_ADDRESS .env | cut -d= -f2)"
