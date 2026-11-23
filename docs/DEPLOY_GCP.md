# Deploying on Google Cloud (1× NVIDIA L4, Mumbai)

Target: **~0.8–1.2 s** from the end of the user's speech to the first audio.

## What runs where

One `g2-standard-8` VM (8 vCPU, 32 GB RAM, 1× L4 24 GB) in `asia-south1` (Mumbai) runs four containers, defined in [`deploy/gcp/docker-compose.yml`](../deploy/gcp/docker-compose.yml):

| Container | What it does | GPU memory |
|---|---|---|
| `svara` | Svara TTS server ([Kenpath/svara-tts-inference](https://github.com/Kenpath/svara-tts-inference)), built by [`svara.Dockerfile`](../deploy/gcp/svara.Dockerfile) on the vLLM image (upstream's Dockerfile no longer builds). Streams PCM audio. | ~45% (~10 GB) |
| `qwen` | Qwen3-1.7B on the vLLM OpenAI-compatible server | ~25% (~6 GB) |
| `agent` | This app: Silero VAD (CPU), Whisper large-v3-turbo (CUDA fp16), WebSocket session | ~2.5 GB |
| `caddy` | Automatic HTTPS. Browsers only allow the microphone on `https://`. | – |

Latency path for one turn:
`VAD silence (300 ms)` → `Whisper (~150–300 ms)` → `Qwen first sentence (~100–200 ms)` → `Svara first audio chunk (~150–300 ms)` → browser.
TTS starts on sentence 1 while Qwen is still generating, and audio plays as it arrives.

## 0. Before you start: GPU quota (required)

Your project currently has **`GPUS_ALL_REGIONS` = 0**. No GPU VM can be created until this is raised.

1. **Free-trial accounts can't use GPUs.** If your billing account is still a free trial, click **Activate full account** (Billing page, or the banner in the console). Your remaining credits are kept.
2. Open **IAM & Admin → Quotas**: `https://console.cloud.google.com/iam-admin/quotas`
   Filter for **"GPUs (all regions)"**, select it, click **Edit**, request **1**, and give a short reason (for example "voice AI inference on 1 L4").
   Approval usually takes minutes to a couple of days.
3. The regional L4 quota in `asia-south1` is already 1, and L4s are offered in zones a, b and c.

## 1. Create the VM (from your Mac)

```bash
bash deploy/gcp/create_vm.sh            # on-demand
SPOT=true bash deploy/gcp/create_vm.sh  # ~60-70% cheaper; Google can stop it anytime
ZONE=asia-south1-b bash deploy/gcp/create_vm.sh   # if zone a has no L4 capacity
```

The script:
- checks the GPU quota
- reserves a static IP
- opens ports 80/443
- creates the VM from Google's Deep Learning image (`common-cu129-ubuntu-2204-nvidia-580`, NVIDIA driver preinstalled)

## 2. Deploy (from your Mac)

```bash
bash deploy/gcp/deploy.sh
```

The script uploads the code, installs Docker and the NVIDIA container toolkit if they're missing, clones the Svara server, and runs `docker compose up -d --build`.

The first start downloads about 25 GB of images and model weights and takes **10–20 minutes**. Follow progress with:

```bash
gcloud compute ssh voice-agent-gpu --zone asia-south1-a --command \
  "sudo docker compose -f ~/voice-agent/deploy/gcp/docker-compose.yml logs -f"
```

When `agent` logs `Preloaded svara-tts-v1 (remote)`, open **`https://<your-ip-with-dashes>.sslip.io`**. `create_vm.sh` prints this URL.

`sslip.io` gives a free hostname for your IP, so Caddy can get a real HTTPS certificate. To use your own domain, point an A record at the IP, set `SITE_ADDRESS=your.domain` in `~/voice-agent/deploy/gcp/.env` on the VM, and re-run `deploy.sh`.

## 3. Check it

```bash
curl https://<host>/health        # all components listed
curl https://<host>/info          # "cuda_available": true, "cuda_device": "NVIDIA L4"
```

The playback mode in the page defaults to **Fast start** on GCP.

## Cost control (important with credits)

Billing runs **while the VM is running**, whether or not anyone uses it.

```bash
gcloud compute instances stop  voice-agent-gpu --zone asia-south1-a   # stop GPU billing
gcloud compute instances start voice-agent-gpu --zone asia-south1-a   # containers auto-restart
```

Approximate costs; check the [pricing calculator](https://cloud.google.com/products/calculator) for current Mumbai prices:
- `g2-standard-8` on-demand: roughly **US$0.85–1.00 per hour**. Spot is roughly 60–70% less.
- When stopped, you still pay for the 150 GB disk (about US$15/month) and the reserved static IP (small).
- With about US$350 (₹30k) of credit, that's roughly **350 GPU-hours on-demand**, or 2–3× more with Spot.
- Free-trial credits expire after 90 days.
- Set a **budget alert** under Billing → Budgets & alerts.

To delete everything:

```bash
gcloud compute instances delete voice-agent-gpu --zone asia-south1-a
gcloud compute addresses delete voice-agent-gpu-ip --region asia-south1
gcloud compute firewall-rules delete allow-voice-agent-web
```

## Tuning

Set these in `deploy/gcp/.env` on the VM, then run `sudo docker compose up -d`.

| Variable | Default | Effect |
|---|---|---|
| `VAD_MIN_SILENCE_DURATION_MS` | 300 | 200 saves ~100 ms but may cut off slow speakers |
| `WHISPER_LANGUAGE` | hi | Fixing the language skips detection |
| `SVARA_GPU_MEMORY_UTILIZATION` | 0.45 | Lower it if `qwen` fails to start with an out-of-memory error |
| `QWEN_GPU_MEMORY_UTILIZATION` | 0.25 | |
| `VLLM_IMAGE` | vllm/vllm-openai:v0.34.0 | Used by both Qwen and Svara (Svara needs vLLM ≥ 0.17) |

## Troubleshooting

- **`qwen` restarts with "Free memory on device ... less than desired GPU memory utilization"**: Svara took more memory than planned. Lower `SVARA_GPU_MEMORY_UTILIZATION` to 0.40.
- **Mic button does nothing**: the page must be opened over `https://`, not `http://<ip>`.
- **`ZONE_RESOURCES_POOL_EXHAUSTED` when creating the VM**: L4 capacity is temporarily used up in that zone. Try `ZONE=asia-south1-b` or `-c`.
- **No audio, and `agent` logs `Svara server error`**: check `sudo docker compose logs svara`. The first start compiles kernels, so wait for its healthcheck to pass.
