#!/usr/bin/env python3
"""Балабол-бот — голосовой бот. Свободный диалог: говори — Балабол-бот отвечает."""
import json
import logging
import math
import os
import re
import select
import struct
import subprocess
import sys
import termios
import time
import tty
import urllib.request
import wave

import pyaudio

# Log everything to file
logging.basicConfig(
    filename="balabol.log",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("balabol")


def lprint(*args, **kwargs):
    """Print to stdout and log to file."""
    text = " ".join(str(a) for a in args)
    log.info(text)
    print(*args, **kwargs)

import argparse
parser = argparse.ArgumentParser(description="Балабол-бот")
parser.add_argument("model", nargs="?", default="qwen2.5:7b", help="Ollama model name")
parser.add_argument("--no-hallucination-filter", action="store_true", help="Disable whisper hallucination filter")
parser.add_argument("--no-echo-cancel", action="store_true", help="Disable echo cancellation (don't listen during playback)")
args = parser.parse_args()

MODEL = args.model
HALLUCINATION_FILTER = not args.no_hallucination_filter
VOICE_MODEL = os.path.expanduser("~/piper-voices/ru_RU-denis-medium.onnx")
PIPER = os.path.expanduser("~/piper-env/bin/piper")
WHISPER_MODEL = os.path.expanduser("~/whisper-models/ggml-medium.bin")
WHISPER_CLI = "whisper-cli"

WAV_IN = "/tmp/balabolka_in.wav"
WAV_OUT = "/tmp/balabolka_out.wav"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Audio settings
RATE = 16000
CHANNELS = 1
CHUNK = 1024  # frames per buffer (~64ms at 16kHz)

# Voice activity detection
RMS_THRESHOLD = 0          # auto-calibrated at startup
SILENCE_TIMEOUT = 1.5      # seconds of silence after speech to stop
MIN_SPEECH_SEC = 0.5       # minimum speech to process
MAX_RECORD_SEC = 20        # safety limit

SYSTEM_PROMPT = """Ты друг. Общайся как друг — просто, по-свойски.
Если спрашивают — объясняй понятно, помогай.

Правила:
- Коротко, 1-3 предложения
- Никаких списков, нумерации, эмодзи, markdown
- Говори как живой человек
- ТОЛЬКО русский язык
"""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
muted = False

import random
HELLO_PHRASES = [
    "Алё? Вас не слышно.",
    "Алё, алё?",
    "Говорите, я слушаю.",
    "Алё? Вы тут?",
    "Не слышу вас, алё!",
    "Алё, вас не слышно, говорите громче.",
]
HEARD_PHRASES = [
    "Да, сейчас слышу.",
    "Ага, теперь слышу, говорите.",
    "О, теперь слышно. Слушаю.",
    "Да-да, слышу вас.",
]
WAIT_BEFORE_HELLO = 5.0  # seconds of silence before "алё"
HELLO_INTERVAL = 6.0     # seconds between "алё" repeats
MAX_HELLOS = float("inf") # no limit


def key_pressed():
    """Check if a key was pressed (non-blocking)."""
    return select.select([sys.stdin], [], [], 0)[0]


def check_mute_toggle():
    """Toggle mute if space was pressed."""
    global muted
    if key_pressed():
        ch = sys.stdin.read(1)
        if ch == " ":
            muted = not muted
            status = "🔇 МЬЮТ" if muted else "🎙 СЛУШАЮ"
            lprint(f"\n  {status}", flush=True)
            return True
    return False


def rms(data):
    """Calculate RMS of audio chunk."""
    count = len(data) // 2
    shorts = struct.unpack(f"<{count}h", data)
    sum_sq = sum(s * s for s in shorts)
    return math.sqrt(sum_sq / count) if count else 0


def listen_for_speech(pa):
    """Listen for speech, record it, stop after silence."""
    global muted
    stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                     rate=RATE, input=True, frames_per_buffer=CHUNK)

    if muted:
        lprint("  🔇 Мьют (пробел чтобы включить)...", end="", flush=True)
        while muted:
            stream.read(CHUNK, exception_on_overflow=False)  # drain mic buffer
            check_mute_toggle()
            time.sleep(0.05)

    lprint("  🎙 Слушаю...", end="", flush=True)

    frames = []
    has_speech = False
    silence_start = None
    speech_start = None
    waiting_since = time.time()
    hello_count = 0
    last_hello_time = 0
    said_hello = False

    try:
        while True:
            # Check for mute toggle
            if check_mute_toggle():
                if muted:
                    stream.stop_stream()
                    stream.close()
                    return 0

            data = stream.read(CHUNK, exception_on_overflow=False)
            level = rms(data)

            if level > RMS_THRESHOLD:
                if not has_speech:
                    has_speech = True
                    speech_start = time.time()
                    if said_hello:
                        # Say "now I hear you" before recording
                        stream.stop_stream()
                        speak(random.choice(HEARD_PHRASES))
                        stream.start_stream()
                    lprint(f" говори...", flush=True)
                silence_start = None
                frames.append(data)
            elif has_speech:
                frames.append(data)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_TIMEOUT:
                    break
            else:
                # No speech yet — check if we should say "алё"
                elapsed = time.time() - waiting_since
                if (not said_hello and elapsed > WAIT_BEFORE_HELLO) or \
                   (said_hello and hello_count < MAX_HELLOS and
                    time.time() - last_hello_time > HELLO_INTERVAL):
                    phrase = random.choice(HELLO_PHRASES)
                    lprint(f"\n  📞 {phrase}", flush=True)
                    stream.stop_stream()
                    speak(phrase)
                    stream.start_stream()
                    said_hello = True
                    hello_count += 1
                    last_hello_time = time.time()
                    lprint("  🎙 Слушаю...", end="", flush=True)

            # Safety: max recording time
            if speech_start and (time.time() - speech_start) > MAX_RECORD_SEC:
                break

    finally:
        stream.stop_stream()
        stream.close()

    if not frames:
        return 0

    # Save to WAV
    with wave.open(WAV_IN, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    duration = len(frames) * CHUNK / RATE
    lprint(f"  📼 Записано: {duration:.1f}с")
    return duration


# Whisper hallucination patterns — common false outputs on noise/silence
WHISPER_HALLUCINATIONS = [
    "субтитр", "корректор", "продолжение следует",
    "подписывайтесь", "спасибо за просмотр", "до новых встреч",
    "аплодисменты", "смех", "www.", "http",
    "subtitle", "thank you", "subscribe", "copyright",
    "режиссёр", "режиссер", "редактор субтитр",
]


def is_hallucination(text):
    """Check if whisper output is a hallucination."""
    lower = text.lower()
    # Too short
    if len(lower) < 3:
        return True
    # Contains hallucination keywords
    for pattern in WHISPER_HALLUCINATIONS:
        if pattern in lower:
            return True
    # Mostly non-speech chars
    letters = sum(1 for c in lower if c.isalpha())
    if letters < len(lower) * 0.5:
        return True
    return False


def transcribe():
    """Transcribe WAV file using whisper.cpp."""
    t0 = time.time()
    result = subprocess.run(
        [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", WAV_IN, "-l", "ru", "--no-timestamps", "-np"],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        lprint(f"  whisper ошибка: {result.stderr[:200]}")
        return ""
    text = result.stdout.strip()
    text = re.sub(r"\[.*?\]", "", text).strip()
    lprint(f"  🎤 ({elapsed:.1f}с): {text}")
    if HALLUCINATION_FILTER and is_hallucination(text):
        lprint("  ⚠ галлюцинация whisper, пропускаю")
        return ""
    return text


_translator = None


def get_translator():
    """Lazy-load argos translator (zh → ru via en)."""
    global _translator
    if _translator is not None:
        return _translator
    try:
        from argostranslate import translate
        installed = translate.get_installed_languages()
        lang_map = {l.code: l for l in installed}
        # Try zh → ru directly
        if "zh" in lang_map and "ru" in lang_map:
            t = lang_map["zh"].get_translation(lang_map["ru"])
            if t:
                _translator = t
                return _translator
        # Fallback: zh → en
        if "zh" in lang_map and "en" in lang_map:
            _translator = lang_map["zh"].get_translation(lang_map["en"])
            return _translator
    except ImportError:
        pass
    _translator = False
    return False


def has_chinese(text):
    """Check if text contains Chinese characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def translate_foreign(text):
    """Translate Chinese parts of text to Russian."""
    if not has_chinese(text):
        return text

    translator = get_translator()
    if not translator:
        # Fallback: just remove Chinese chars
        return re.sub(r"[^\sа-яА-ЯёЁa-zA-Z0-9.,!?;:\-\(\)\"']+", "", text).strip()

    # Split into parts: Russian/Latin vs Chinese
    parts = re.split(r"([\u4e00-\u9fff，。！？；：、]+)", text)
    result = []
    for part in parts:
        if has_chinese(part):
            translated = translator.translate(part)
            result.append(translated)
            lprint(f"\n  🔄 перевод: {part} → {translated}", end="", flush=True)
        else:
            result.append(part)
    return "".join(result).strip()


def generate_tts(text, wav_path):
    """Generate TTS audio file from text. Returns path."""
    clean = text.replace("*", "").replace("#", "").replace("`", "")
    clean = re.sub(r"[^\sа-яА-ЯёЁa-zA-Z0-9.,!?;:\-\(\)\"']+", "", clean).strip()
    if not clean:
        return None
    subprocess.run(
        [PIPER, "--model", VOICE_MODEL, "--output_file", wav_path],
        input=clean.encode(),
        capture_output=True,
    )
    return wav_path


def ask_and_speak(user_text, pa=None):
    """Stream LLM response and speak sentence by sentence."""
    messages.append({"role": "user", "content": user_text})
    data = json.dumps({"model": MODEL, "messages": messages, "stream": True}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})

    reply = ""
    buffer = ""
    first_token_time = None
    sentence_num = 0
    player = None

    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                if first_token_time is None:
                    first_token_time = time.time()
                    lprint("  Балабол-бот: ", end="", flush=True)
                # Print only non-Chinese characters
                display = re.sub(r"[\u4e00-\u9fff，。！？；：、]+", "", token)
                if display:
                    print(display, end="", flush=True)
                reply += token
                buffer += token

                # Check if we have a complete sentence
                if re.search(r"[.!?]\s", buffer) or buffer.endswith((".", "!", "?")):
                    # Speak this sentence while LLM continues generating
                    sentence = translate_foreign(buffer.strip())
                    if sentence:
                        wav_path = f"/tmp/balabolka_sent_{sentence_num}.wav"
                        generate_tts(sentence, wav_path)
                        # Wait for previous sentence to finish
                        if player and player.poll() is None:
                            player.wait()
                        player = subprocess.Popen(["afplay", wav_path])
                        sentence_num += 1
                    buffer = ""

            if chunk.get("done"):
                break

    # Speak remaining buffer
    if buffer.strip():
        sentence = translate_foreign(buffer.strip())
        if sentence:
            wav_path = f"/tmp/balabolka_sent_{sentence_num}.wav"
            generate_tts(sentence, wav_path)
            if player and player.poll() is None:
                player.wait()
            player = subprocess.Popen(["afplay", wav_path])

    # Wait for last sentence to finish
    if player and player.poll() is None:
        player.wait()

    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
    reply = translate_foreign(reply)
    log.info(f"LLM ответ: {reply}")
    messages.append({"role": "assistant", "content": reply})
    return reply, first_token_time


WAV_MIC_DURING = "/tmp/balabolka_mic_during.wav"
WAV_CLEANED = "/tmp/balabolka_cleaned.wav"


def echo_cancel(mic_wav, bot_wav, out_wav):
    """Remove bot's voice from mic recording using cross-correlation alignment."""
    import numpy as np
    from scipy.signal import fftconvolve

    # Read mic recording
    with wave.open(mic_wav, "rb") as wf:
        mic_rate = wf.getframerate()
        mic_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)

    # Read bot audio (what was played)
    with wave.open(bot_wav, "rb") as wf:
        bot_rate = wf.getframerate()
        bot_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)

    # Resample bot audio to mic rate if needed
    if bot_rate != mic_rate:
        from scipy.signal import resample
        bot_data = resample(bot_data, int(len(bot_data) * mic_rate / bot_rate))

    # Ensure bot_data is not longer than mic_data
    if len(bot_data) > len(mic_data):
        bot_data = bot_data[:len(mic_data)]

    # Find alignment offset using cross-correlation
    correlation = fftconvolve(mic_data, bot_data[::-1], mode="full")
    offset = int(np.argmax(np.abs(correlation)) - len(bot_data) + 1)
    offset = max(0, min(offset, len(mic_data) - 1))

    # Align and scale bot signal to match mic level
    aligned_bot = np.zeros_like(mic_data)
    end = min(offset + len(bot_data), len(mic_data))
    aligned_bot[offset:end] = bot_data[:end - offset]

    # Find optimal scaling factor (least squares)
    if np.dot(aligned_bot, aligned_bot) > 0:
        scale = np.dot(mic_data, aligned_bot) / np.dot(aligned_bot, aligned_bot)
        scale = max(0, min(scale, 5.0))  # clamp
    else:
        scale = 0

    # Subtract scaled bot audio from mic
    cleaned = mic_data - scale * aligned_bot

    # Check if there's meaningful speech remaining
    rms_before = np.sqrt(np.mean(mic_data ** 2))
    rms_after = np.sqrt(np.mean(cleaned ** 2))
    # Ratio: how much was removed. If <50% removed, likely no user speech — just noise residual
    reduction = 1 - (rms_after / rms_before) if rms_before > 0 else 0
    log.info(f"Echo cancel: offset={offset}, scale={scale:.2f}, rms_before={rms_before:.0f}, rms_after={rms_after:.0f}, reduction={reduction:.1%}")

    # Normalize and save
    cleaned = np.clip(cleaned, -32768, 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(mic_rate)
        wf.writeframes(cleaned.tobytes())

    # User speech detected only if: significant signal remains AND reduction was meaningful
    # (if reduction < 20%, echo cancel barely changed anything — it's just noise, not bot+user)
    return rms_after > RMS_THRESHOLD * 5 and reduction > 0.3


def speak(text, pa=None):
    clean = text.replace("*", "").replace("#", "").replace("`", "")
    t0 = time.time()
    subprocess.run(
        [PIPER, "--model", VOICE_MODEL, "--output_file", WAV_OUT],
        input=clean.encode(),
        capture_output=True,
    )
    tts_time = time.time() - t0
    lprint(f"  🔊 голос: {tts_time:.1f}с")
    subprocess.run(["pkill", "-f", "afplay.*balabolka"], capture_output=True)

    # Play audio while recording mic simultaneously for echo cancellation
    player = subprocess.Popen(["afplay", WAV_OUT])

    if pa:
        stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                         rate=RATE, input=True, frames_per_buffer=CHUNK)
        mic_frames = []

        while player.poll() is None:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                mic_frames.append(data)
            except Exception:
                break

        # Keep recording for a short time after playback ends (user might still be talking)
        extra_time = time.time()
        while time.time() - extra_time < SILENCE_TIMEOUT:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                mic_frames.append(data)
                level = rms(data)
                if level < RMS_THRESHOLD:
                    break
            except Exception:
                break

        stream.stop_stream()
        stream.close()

        if mic_frames:
            # Save mic recording
            with wave.open(WAV_MIC_DURING, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(RATE)
                wf.writeframes(b"".join(mic_frames))

            # Echo cancellation — remove bot voice, keep user voice
            has_speech = echo_cancel(WAV_MIC_DURING, WAV_OUT, WAV_CLEANED)

            if has_speech:
                lprint("  ⚡ Речь обнаружена во время озвучки!")
                # Copy cleaned audio to WAV_IN for transcription
                subprocess.run(["cp", WAV_CLEANED, WAV_IN], capture_output=True)
                return "interrupted"
    else:
        player.wait()

    return "done"


def calibrate(pa):
    """Measure background noise for 2 seconds and set threshold."""
    global RMS_THRESHOLD
    lprint("  🔇 Калибровка микрофона (2 сек тишины)...", end="", flush=True)
    stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                     rate=RATE, input=True, frames_per_buffer=CHUNK)
    levels = []
    for _ in range(int(RATE / CHUNK * 2)):  # 2 seconds
        data = stream.read(CHUNK, exception_on_overflow=False)
        levels.append(rms(data))
    stream.stop_stream()
    stream.close()

    avg = sum(levels) / len(levels)
    mx = max(levels)
    RMS_THRESHOLD = max(int(mx * 1.8), int(avg * 3), 40)
    lprint(f" шум={avg:.0f}, пик={mx:.0f}, порог={RMS_THRESHOLD}")


def main():
    lprint("=== Балабол-бот (свободный диалог) ===")
    lprint(f"Модель: {MODEL}")
    lprint("Просто говори — Балабол-бот ответит.")
    lprint("Пробел = мьют/размьют | Ctrl+C = выход")
    lprint()

    pa = pyaudio.PyAudio()
    calibrate(pa)
    lprint()

    # Set terminal to raw mode for non-blocking key reads
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while True:
            duration = listen_for_speech(pa)

            if duration < MIN_SPEECH_SEC:
                continue

            text = transcribe()
            if not text:
                continue

            t0 = time.time()
            lprint("  💭 Думаю...", end="\r", flush=True)
            try:
                reply, first_token_time = ask_and_speak(text)
            except Exception as e:
                lprint(f"Ошибка: {e}")
                continue

            t1 = time.time()
            ttft = first_token_time - t0 if first_token_time else 0
            total = t1 - t0
            lprint(f"\n  ⏱ думал {ttft:.1f}с | ответ {total:.1f}с")

            lprint()

    except KeyboardInterrupt:
        lprint("\nПока!")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        pa.terminate()


if __name__ == "__main__":
    main()
