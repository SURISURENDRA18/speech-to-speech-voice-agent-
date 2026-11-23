"""Generate the animated SVGs used by README.md (self-contained, no external assets)."""

import math
import random
import sys
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(sys.argv[1])
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

BG = """
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#0B1020"/>
    <stop offset="0.55" stop-color="#1A1442"/>
    <stop offset="1" stop-color="#0C2A44"/>
  </linearGradient>
  <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#8B5CF6"/>
    <stop offset="0.5" stop-color="#22D3EE"/>
    <stop offset="1" stop-color="#F59E0B"/>
  </linearGradient>
"""


# ----------------------------------------------------------------------------- banner
def banner() -> str:
    W, H = 1280, 360
    rnd = random.Random(7)
    bars = []
    n = 72
    gap = W / n
    for i in range(n):
        x = i * gap + gap * 0.2
        w = gap * 0.6
        # envelope: taller in the middle
        env = 0.35 + 0.65 * math.sin(math.pi * (i + 0.5) / n)
        h = 46 * env
        dur = 0.8 + rnd.random() * 0.9
        delay = -rnd.random() * 2
        bars.append(
            f'<rect class="bar" x="{x:.1f}" y="{H - 18 - h:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{w / 2:.1f}" '
            f'style="animation-duration:{dur:.2f}s;animation-delay:{delay:.2f}s"/>'
        )

    greetings = [
        ("नमस्ते", 120, 90, 0), ("வணக்கம்", 1110, 80, 2), ("నమస్కారం", 190, 250, 4),
        ("নমস্কার", 1060, 240, 6), ("Hello", 70, 175, 8), ("नमस्कार", 1185, 165, 10),
        ("ನಮಸ್ಕಾರ", 300, 150, 12), ("સ્વાગત", 960, 140, 14),
    ]
    floats = "\n".join(
        f'<text class="greet" x="{x}" y="{y}" style="animation-delay:{d}s">{escape(t)}</text>'
        for t, x, y, d in greetings
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Speech-to-Speech Voice Agent">
<defs>{BG}
  <radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0" stop-color="#22D3EE" stop-opacity="0.55"/>
    <stop offset="1" stop-color="#22D3EE" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="title" x1="0" y1="0" x2="1" y2="0" gradientUnits="objectBoundingBox">
    <stop offset="0" stop-color="#C4B5FD"/>
    <stop offset="0.5" stop-color="#67E8F9"/>
    <stop offset="1" stop-color="#FCD34D"/>
    <animateTransform attributeName="gradientTransform" type="translate" values="-0.3 0; 0.3 0; -0.3 0" dur="8s" repeatCount="indefinite"/>
  </linearGradient>
  <clipPath id="frame"><rect width="{W}" height="{H}" rx="22"/></clipPath>
</defs>
<style>
  .bar {{ fill: url(#accent); opacity: .85; transform-box: fill-box; transform-origin: 50% 100%; animation: eq 1.2s ease-in-out infinite alternate; }}
  @keyframes eq {{ 0% {{ transform: scaleY(.15) }} 100% {{ transform: scaleY(1) }} }}
  .ring {{ fill: none; stroke: #67E8F9; stroke-width: 2; transform-box: fill-box; transform-origin: center; animation: ripple 3s ease-out infinite; opacity: 0; }}
  .ring.r2 {{ animation-delay: 1s }} .ring.r3 {{ animation-delay: 2s }}
  @keyframes ripple {{ 0% {{ transform: scale(1); opacity: .8 }} 100% {{ transform: scale(2.8); opacity: 0 }} }}
  .greet {{ font: 600 22px {FONT}; fill: #C7D2FE; opacity: 0; animation: drift 16s ease-in-out infinite; }}
  @keyframes drift {{ 0% {{ opacity: 0; transform: translateY(10px) }} 6% {{ opacity: .55 }} 20% {{ opacity: .55 }} 28% {{ opacity: 0; transform: translateY(-14px) }} 100% {{ opacity: 0 }} }}
  .t {{ font: 800 56px {FONT}; letter-spacing: -1px; }}
  .s {{ font: 500 21px {FONT}; fill: #A5B4D4; }}
  .orb {{ animation: breathe 3s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }}
  @keyframes breathe {{ 0%,100% {{ transform: scale(1) }} 50% {{ transform: scale(1.06) }} }}
</style>
<g clip-path="url(#frame)">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <circle cx="640" cy="98" r="120" fill="url(#glow)"/>
  {floats}
  <circle class="ring" cx="640" cy="98" r="34"/>
  <circle class="ring r2" cx="640" cy="98" r="34"/>
  <circle class="ring r3" cx="640" cy="98" r="34"/>
  <g class="orb">
    <circle cx="640" cy="98" r="36" fill="url(#accent)"/>
    <rect x="631" y="76" width="18" height="30" rx="9" fill="#0B1020"/>
    <path d="M622 96 a18 18 0 0 0 36 0" fill="none" stroke="#0B1020" stroke-width="4" stroke-linecap="round"/>
    <line x1="640" y1="114" x2="640" y2="122" stroke="#0B1020" stroke-width="4" stroke-linecap="round"/>
  </g>
  <text class="t" x="640" y="212" text-anchor="middle" fill="url(#title)">Speech-to-Speech Voice Agent</text>
  <text class="s" x="640" y="252" text-anchor="middle">Real-time, full-duplex voice AI for Indian languages</text>
  {''.join(bars)}
</g>
</svg>
"""


# ----------------------------------------------------------------------------- typing
def typing(color: str = "#67E8F9", cursor: str = "#F59E0B") -> str:
    lines = [
        "Speak Hindi, Tamil, Telugu, Bengali... it answers back.",
        "Silero VAD -> Whisper -> Qwen3 -> Svara TTS",
        "Interrupt anytime. It stops talking and listens.",
        "Runs on an 8 GB Mac or a single NVIDIA L4.",
    ]
    W, H, fs = 1280, 64, 24
    cw = fs * 0.6
    slot = 4.5  # seconds per line
    total = slot * len(lines)
    parts = []
    for i, line in enumerate(lines):
        n = len(line)
        tw = n * cw
        x0 = (W - tw) / 2
        t0 = i * slot
        type_end = t0 + 2.2
        hold_end = t0 + 3.7
        del_end = t0 + 4.2
        # build discrete keyframes for characters typed and deleted
        times, vals = [0.0], [0]
        for k in range(1, n + 1):
            times.append(t0 + (type_end - t0) * k / n)
            vals.append(k)
        for k in range(n - 1, -1, -1):
            times.append(hold_end + (del_end - hold_end) * (n - k) / n)
            vals.append(k)
        times.append(del_end + 0.05)
        vals.append(0)
        times.append(total)
        vals.append(0)
        if t0 > 0:
            times.insert(1, t0)
            vals.insert(1, 0)
        kt = ";".join(f"{t / total:.4f}" for t in times)
        wv = ";".join(f"{v * cw:.1f}" for v in vals)
        cx = ";".join(f"{x0 + v * cw + 2:.1f}" for v in vals)
        cop = ";".join("1" if (t0 <= t < del_end + 0.01) else "0" for t in times)
        parts.append(f"""
  <clipPath id="c{i}"><rect x="{x0:.1f}" y="0" height="{H}" width="0">
    <animate attributeName="width" values="{wv}" keyTimes="{kt}" dur="{total}s" calcMode="discrete" repeatCount="indefinite"/>
  </rect></clipPath>
  <text x="{x0:.1f}" y="41" textLength="{tw:.1f}" lengthAdjust="spacingAndGlyphs" clip-path="url(#c{i})" class="ty">{escape(line)}</text>
  <rect class="cur" y="18" width="3" height="30" x="{x0:.1f}" opacity="0">
    <animate attributeName="x" values="{cx}" keyTimes="{kt}" dur="{total}s" calcMode="discrete" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="{cop}" keyTimes="{kt}" dur="{total}s" calcMode="discrete" repeatCount="indefinite"/>
  </rect>""")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{escape(' / '.join(lines))}">
<defs>{BG}</defs>
<style>
  .ty {{ font: 600 {fs}px {MONO}; fill: {color}; }}
  .cur {{ fill: {cursor}; }}
</style>
{''.join(parts)}
</svg>
"""


# ----------------------------------------------------------------------------- pipeline
def pipeline() -> str:
    W, H = 1280, 330
    cy = 170
    stages = [
        ("01", "Silero VAD", "detects end of turn", "~300 ms silence"),
        ("02", "Whisper", "large-v3-turbo STT", "150–300 ms"),
        ("03", "Qwen3 1.7B", "streaming replies", "100–200 ms"),
        ("04", "Indic Norm", "₹500 → spoken words", "rule / WFST"),
        ("05", "Svara TTS", "24 kHz Indic voices", "150–300 ms"),
    ]
    xs = [250, 445, 640, 835, 1030]
    bw, bh = 168, 104
    start, end = 70, 1210
    dur = 5.0
    travel = 0.8  # fraction of the loop spent travelling
    boxes = []
    for (num, title, sub, lat), x in zip(stages, xs):
        frac = (x - start) / (end - start) * travel
        delay = frac * dur - 0.35
        boxes.append(f"""
  <g>
    <rect class="box" x="{x - bw / 2}" y="{cy - bh / 2}" width="{bw}" height="{bh}" rx="16"/>
    <rect class="lit" x="{x - bw / 2}" y="{cy - bh / 2}" width="{bw}" height="{bh}" rx="16" style="animation-delay:{delay:.2f}s"/>
    <text class="num" x="{x - bw / 2 + 16}" y="{cy - bh / 2 + 24}">{num}</text>
    <text class="bt" x="{x}" y="{cy + 4}" text-anchor="middle">{escape(title)}</text>
    <text class="bs" x="{x}" y="{cy + 28}" text-anchor="middle">{escape(sub)}</text>
    <rect class="chip" x="{x - 62}" y="{cy + bh / 2 + 16}" width="124" height="26" rx="13"/>
    <text class="ct" x="{x}" y="{cy + bh / 2 + 33.5}" text-anchor="middle">{escape(lat)}</text>
  </g>""")
    kt = f"0;{travel};1"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Pipeline: microphone, Silero VAD, Whisper, Qwen3, Indic normalization, Svara TTS, speaker">
<defs>{BG}
  <filter id="blur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6"/></filter>
  <clipPath id="frame"><rect width="{W}" height="{H}" rx="22"/></clipPath>
</defs>
<style>
  .box {{ fill: #121A33; stroke: #2A3560; stroke-width: 1.5; }}
  .lit {{ fill: none; stroke: url(#accent); stroke-width: 3; opacity: 0; animation: lit {dur}s linear infinite; }}
  @keyframes lit {{ 0% {{ opacity: 0 }} 4% {{ opacity: 1 }} 16% {{ opacity: 1 }} 24% {{ opacity: 0 }} 100% {{ opacity: 0 }} }}
  .num {{ font: 700 12px {MONO}; fill: #8B5CF6; }}
  .bt {{ font: 700 19px {FONT}; fill: #F1F5F9; }}
  .bs {{ font: 500 13px {FONT}; fill: #94A3C4; }}
  .chip {{ fill: #0F2236; stroke: #1E4A63; }}
  .ct {{ font: 600 12px {MONO}; fill: #67E8F9; }}
  .wire {{ stroke: #2A3560; stroke-width: 2; stroke-dasharray: 6 6; animation: flow 1s linear infinite; }}
  @keyframes flow {{ to {{ stroke-dashoffset: -12 }} }}
  .end {{ fill: #121A33; stroke: url(#accent); stroke-width: 2.5; }}
  .lbl {{ font: 600 13px {FONT}; fill: #A5B4D4; }}
  .h {{ font: 700 15px {FONT}; fill: #E2E8F0; letter-spacing: 2px; }}
  .barge {{ fill: none; stroke: #F43F5E; stroke-width: 2; stroke-dasharray: 5 7; animation: flow2 1.4s linear infinite; }}
  @keyframes flow2 {{ to {{ stroke-dashoffset: 24 }} }}
  .bl {{ font: 600 13px {FONT}; fill: #FDA4AF; }}
  .wave {{ fill: none; stroke: #67E8F9; stroke-width: 2.5; stroke-linecap: round; animation: pulse 1s ease-in-out infinite alternate; }}
  .wave.w2 {{ animation-delay: .3s }}
  @keyframes pulse {{ from {{ opacity: .25 }} to {{ opacity: 1 }} }}
</style>
<g clip-path="url(#frame)">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <text class="h" x="40" y="44">HOW A TURN FLOWS</text>
  <text class="lbl" x="{W - 40}" y="44" text-anchor="end">target: ~0.8–1.2 s to first audio on one NVIDIA L4</text>

  <line class="wire" x1="112" y1="{cy}" x2="1168" y2="{cy}"/>

  <path class="barge" d="M 70 {cy - 46} C 70 30, 1030 30, 1030 {cy - bh / 2 - 4}"/>
  <text class="bl" x="560" y="96" text-anchor="middle">barge-in: new speech cancels the LLM + TTS instantly</text>

  <circle class="end" cx="{start}" cy="{cy}" r="42"/>
  <rect x="61" y="{cy - 22}" width="18" height="30" rx="9" fill="#67E8F9"/>
  <path d="M52 {cy - 2} a18 18 0 0 0 36 0" fill="none" stroke="#67E8F9" stroke-width="3.5" stroke-linecap="round"/>
  <line x1="70" y1="{cy + 16}" x2="70" y2="{cy + 24}" stroke="#67E8F9" stroke-width="3.5" stroke-linecap="round"/>
  <text class="lbl" x="{start}" y="{cy + 72}" text-anchor="middle">you speak</text>

  <circle class="end" cx="{end}" cy="{cy}" r="42"/>
  <path d="M1188 {cy - 10} h10 l14 -12 v44 l-14 -12 h-10 z" fill="#67E8F9"/>
  <path class="wave" d="M1220 {cy - 9} q8 9 0 18"/>
  <path class="wave w2" d="M1227 {cy - 16} q14 16 0 32"/>
  <text class="lbl" x="{end}" y="{cy + 72}" text-anchor="middle">agent replies</text>

  {''.join(boxes)}

  <g>
    <circle r="14" cy="{cy}" fill="#22D3EE" opacity=".6" filter="url(#blur)">
      <animate attributeName="cx" values="{start};{end};{end}" keyTimes="{kt}" dur="{dur}s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values=".6;.6;0" keyTimes="{kt}" dur="{dur}s" repeatCount="indefinite"/>
    </circle>
    <circle r="6" cy="{cy}" fill="#F0FDFF">
      <animate attributeName="cx" values="{start};{end};{end}" keyTimes="{kt}" dur="{dur}s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="1;1;0" keyTimes="{kt}" dur="{dur}s" repeatCount="indefinite"/>
    </circle>
  </g>
</g>
</svg>
"""


# ----------------------------------------------------------------------------- footer
def footer() -> str:
    W, H = 1280, 140

    def wave(amp: float, length: float, y: float) -> str:
        pts = []
        for i in range(0, int(W * 2) + 1, 8):
            pts.append(f"{i},{y + amp * math.sin(2 * math.pi * i / length):.1f}")
        return "M" + " L".join(pts)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Thanks for visiting">
<defs>{BG}<clipPath id="frame"><rect width="{W}" height="{H}" rx="22"/></clipPath></defs>
<style>
  .w {{ fill: none; stroke-width: 2.5; stroke-linecap: round; animation: slide 6s linear infinite; }}
  .w2 {{ animation-duration: 9s; opacity: .55 }}
  .w3 {{ animation-duration: 13s; opacity: .3 }}
  @keyframes slide {{ to {{ transform: translateX(-640px) }} }}
  .f {{ font: 700 22px {FONT}; fill: #E2E8F0; }}
  .g {{ font: 500 15px {FONT}; fill: #94A3C4; }}
</style>
<g clip-path="url(#frame)">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <path class="w" stroke="url(#accent)" d="{wave(14, 320, 104)}"/>
  <path class="w w2" stroke="#8B5CF6" d="{wave(20, 213.33, 108)}"/>
  <path class="w w3" stroke="#22D3EE" d="{wave(26, 640, 112)}"/>
  <text class="f" x="640" y="48" text-anchor="middle">Speak in your language. Hear it answer back.</text>
  <text class="g" x="640" y="74" text-anchor="middle">If this project helps you, consider giving it a ★</text>
</g>
</svg>
"""


OUT.mkdir(parents=True, exist_ok=True)
for name, fn in [("banner", banner), ("typing-dark", typing), ("typing-light", lambda: typing("#0E7490", "#D97706")), ("pipeline", pipeline), ("footer", footer)]:
    (OUT / f"{name}.svg").write_text(fn(), encoding="utf-8")
    print("wrote", OUT / f"{name}.svg")
