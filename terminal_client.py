"""Terminal client for the Speech-to-Speech Voice Agent.

Talks to ws://<host>:<port>/ws/voice: streams microphone (or a WAV file) as
16 kHz PCM16, prints transcripts / LLM text, and plays the TTS reply.

Usage:
    python terminal_client.py                     # talk through your mic
    python terminal_client.py --wav sample.wav    # send a file instead of mic
    python terminal_client.py --full-duplex       # keep mic open while the agent speaks (use headphones)
"""

import argparse
import asyncio
import base64
import json
import sys
import threading
import wave

import numpy as np
import sounddevice as sd
import websockets

IN_RATE = 16000
BLOCK = 512  # 32 ms at 16 kHz, what Silero VAD expects


class Player:
    """Thread-safe PCM16 playback buffer feeding a sounddevice output stream."""

    def __init__(self, rate: int) -> None:
        self.rate = rate
        self._buf = bytearray()
        self._lock = threading.Lock()
        self.stream = sd.RawOutputStream(
            samplerate=rate, channels=1, dtype="int16", callback=self._callback
        )

    def _callback(self, outdata, frames, _time, _status) -> None:
        need = frames * 2
        with self._lock:
            chunk = bytes(self._buf[:need])
            del self._buf[:need]
        outdata[: len(chunk)] = chunk
        outdata[len(chunk):] = b"\x00" * (need - len(chunk))

    def add(self, pcm: bytes) -> None:
        with self._lock:
            self._buf.extend(pcm)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    @property
    def playing(self) -> bool:
        with self._lock:
            return len(self._buf) > 0


def load_wav_16k(path: str) -> np.ndarray:
    """Load a WAV file as mono int16 at 16 kHz."""
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != IN_RATE:
        n_out = int(len(mono) * IN_RATE / rate)
        mono = np.interp(
            np.linspace(0, len(mono) - 1, n_out), np.arange(len(mono)), mono
        )
    return (np.clip(mono, -1, 1) * 32767).astype(np.int16)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default="ws://localhost:8000/ws/voice")
    ap.add_argument("--wav", help="Send this WAV file instead of the microphone, then exit")
    ap.add_argument("--language", default="hi")
    ap.add_argument("--voice", default="hi_female")
    ap.add_argument("--full-duplex", action="store_true",
                    help="Keep mic open while the agent speaks (barge-in). Use headphones.")
    ap.add_argument("--save", help="Also write the agent's reply audio to this WAV file")
    ap.add_argument("--no-play", action="store_true", help="Don't play reply audio")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    loop = asyncio.get_running_loop()
    mic_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    turn_done = asyncio.Event()
    saved = bytearray()
    out_rate = 24000
    player: Player | None = None if args.no_play else Player(out_rate)
    if player:
        player.stream.start()

    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "start",
            "config": {"sample_rate": IN_RATE, "channels": 1,
                       "language": args.language, "voice": args.voice},
        }))

        async def receiver() -> None:
            nonlocal player, out_rate
            async for raw in ws:
                if isinstance(raw, bytes):  # binary TTS frame, if server sends them
                    saved.extend(raw)
                    if player:
                        player.add(raw)
                    continue
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "connected":
                    print(f"· connected (session {msg['session_id'][:8]})")
                elif t == "session_started":
                    print("· session started — speak now (Ctrl+C to quit)\n")
                elif t == "vad_speech_start":
                    print("🎙  listening…", flush=True)
                elif t == "vad_speech_end":
                    print(f"   (speech ended, {msg.get('duration_ms', 0)} ms)", flush=True)
                elif t == "transcript":
                    print(f"\nYou:   {msg['text']}")
                    print("Agent: ", end="", flush=True)
                elif t == "llm_chunk":
                    print(msg["delta"], end="", flush=True)
                elif t == "tts_chunk":
                    rate = msg.get("sample_rate", 24000)
                    if player and rate != out_rate:
                        player.stream.stop()
                        out_rate = rate
                        player = Player(rate)
                        player.stream.start()
                    pcm = base64.b64decode(msg["audio"])
                    saved.extend(pcm)
                    if player:
                        player.add(pcm)
                elif t == "interrupted":
                    if player:
                        player.clear()
                    print(f"\n   ⏹  interrupted ({msg.get('reason')})")
                elif t == "turn_complete":
                    print("\n")
                    turn_done.set()
                elif t == "error":
                    print(f"\n⚠️  {msg.get('code')}: {msg.get('message')}")
                elif t == "session_stopped":
                    print("· session stopped")

        async def sender() -> None:
            silence = b"\x00" * (BLOCK * 2)
            while True:
                chunk = await mic_q.get()
                # Half-duplex by default: send silence while the agent talks so
                # the speakers don't feed back into the mic and trigger barge-in.
                if player and player.playing and not args.full_duplex:
                    chunk = silence
                await ws.send(chunk)

        recv_task = asyncio.create_task(receiver())
        send_task = asyncio.create_task(sender())

        try:
            if args.wav:
                pcm = load_wav_16k(args.wav)
                pcm = np.concatenate([pcm, np.zeros(IN_RATE * 2, dtype=np.int16)])  # trailing silence
                print(f"· sending {args.wav} ({len(pcm) / IN_RATE:.1f}s)")
                for i in range(0, len(pcm), BLOCK):
                    await mic_q.put(pcm[i:i + BLOCK].tobytes())
                    await asyncio.sleep(BLOCK / IN_RATE)  # real-time pacing
                await asyncio.wait_for(turn_done.wait(), timeout=180)
                while player and player.playing:
                    await asyncio.sleep(0.1)
            else:
                def on_audio(indata, _frames, _time, status) -> None:
                    if status:
                        print(status, file=sys.stderr)
                    try:
                        loop.call_soon_threadsafe(mic_q.put_nowait, bytes(indata))
                    except asyncio.QueueFull:
                        pass

                with sd.RawInputStream(samplerate=IN_RATE, channels=1, dtype="int16",
                                       blocksize=BLOCK, callback=on_audio):
                    await recv_task
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        except TimeoutError:
            print("\n⚠️  timed out waiting for turn_complete")
        finally:
            try:
                await ws.send(json.dumps({"type": "stop"}))
            except Exception:  # noqa: BLE001
                pass
            send_task.cancel()
            recv_task.cancel()
            if player:
                player.stream.stop()
            if args.save and saved:
                with wave.open(args.save, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(out_rate)
                    w.writeframes(bytes(saved))
                print(f"· saved reply audio to {args.save}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye")
