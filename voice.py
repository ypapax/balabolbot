#!/usr/bin/env python3
"""Балабол-бот — голосовой бот. Свободный диалог: говори — Балабол-бот отвечает."""
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
import urllib.request
import wave

import pyaudio

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
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

SYSTEM_PROMPT = """Ты обычный человек, тебе звонят на телефон. Ты дома, занимаешься своими делами.
Ты НЕ ассистент, НЕ бот, НЕ служба поддержки. Ты просто живой человек.
Если тебе назвали имя — это твоё имя. Отвечай от лица этого человека.

Правила:
- Максимум 1-2 коротких предложения
- Никаких списков, нумерации, эмодзи, markdown
- Говори как живой человек по телефону, можно мычать, переспрашивать
- ТОЛЬКО русский язык
"""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def rms(data):
    """Calculate RMS of audio chunk."""
    count = len(data) // 2
    shorts = struct.unpack(f"<{count}h", data)
    sum_sq = sum(s * s for s in shorts)
    return math.sqrt(sum_sq / count) if count else 0


def listen_for_speech(pa):
    """Listen for speech, record it, stop after silence."""
    stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                     rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("  🎙 Слушаю...", end="", flush=True)

    frames = []
    has_speech = False
    silence_start = None
    speech_start = None

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            level = rms(data)

            if level > RMS_THRESHOLD:
                if not has_speech:
                    has_speech = True
                    speech_start = time.time()
                    print(f" говори...", flush=True)
                silence_start = None
                frames.append(data)
            elif has_speech:
                frames.append(data)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_TIMEOUT:
                    break

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
    print(f"  📼 Записано: {duration:.1f}с")
    return duration


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
        print(f"  whisper ошибка: {result.stderr[:200]}")
        return ""
    text = result.stdout.strip()
    text = re.sub(r"\[.*?\]", "", text).strip()
    print(f"  🎤 ({elapsed:.1f}с): {text}")
    return text


def ask_ollama_stream(user_text):
    messages.append({"role": "user", "content": user_text})
    data = json.dumps({"model": MODEL, "messages": messages, "stream": True}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})

    reply = ""
    first_token_time = None
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                if first_token_time is None:
                    first_token_time = time.time()
                    print("  Балабол-бот: ", end="", flush=True)
                print(token, end="", flush=True)
                reply += token
            if chunk.get("done"):
                break

    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
    # Remove non-Russian/non-punctuation characters (Chinese, etc.)
    reply = re.sub(r"[^\sа-яА-ЯёЁa-zA-Z0-9.,!?;:\-\(\)\"']+", "", reply).strip()
    messages.append({"role": "assistant", "content": reply})
    return reply, first_token_time


def speak(text):
    clean = text.replace("*", "").replace("#", "").replace("`", "")
    t0 = time.time()
    subprocess.run(
        [PIPER, "--model", VOICE_MODEL, "--output_file", WAV_OUT],
        input=clean.encode(),
        capture_output=True,
    )
    tts_time = time.time() - t0
    print(f"  🔊 голос: {tts_time:.1f}с")
    subprocess.run(["pkill", "-f", "afplay.*balabolka"], capture_output=True)
    # Wait for playback to finish before listening again
    subprocess.run(["afplay", WAV_OUT])


def calibrate(pa):
    """Measure background noise for 2 seconds and set threshold."""
    global RMS_THRESHOLD
    print("  🔇 Калибровка микрофона (2 сек тишины)...", end="", flush=True)
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
    print(f" шум={avg:.0f}, пик={mx:.0f}, порог={RMS_THRESHOLD}")


def main():
    print("=== Балабол-бот (свободный диалог) ===")
    print(f"Модель: {MODEL}")
    print("Просто говори — Балабол-бот ответит. Ctrl+C для выхода.")
    print()

    pa = pyaudio.PyAudio()
    calibrate(pa)
    print()

    try:
        while True:
            duration = listen_for_speech(pa)

            if duration < MIN_SPEECH_SEC:
                continue

            text = transcribe()
            if not text:
                continue

            t0 = time.time()
            print("  💭 Думаю...", end="\r", flush=True)
            try:
                reply, first_token_time = ask_ollama_stream(text)
            except Exception as e:
                print(f"Ошибка: {e}")
                continue

            t1 = time.time()
            ttft = first_token_time - t0 if first_token_time else 0
            total = t1 - t0
            print(f"\n  ⏱ думал {ttft:.1f}с | ответ {total:.1f}с")

            speak(reply)
            print()

    except KeyboardInterrupt:
        print("\nПока!")
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
