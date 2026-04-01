#!/usr/bin/env python3
"""Балаболка — голосовой бот. Говори в микрофон, Денис отвечает голосом."""
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
VOICE_MODEL = os.path.expanduser("~/piper-voices/ru_RU-denis-medium.onnx")
PIPER = os.path.expanduser("~/piper-env/bin/piper")
WHISPER_MODEL = os.path.expanduser("~/whisper-models/ggml-medium.bin")
WHISPER_CLI = "whisper-cli"

WAV_IN = "/tmp/balabolka_in.wav"
WAV_OUT = "/tmp/balabolka_out.wav"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """Ты голосовой ассистент. Это устный разговор, не чат.
Правила:
- Максимум 1-2 коротких предложения
- Никаких списков, нумерации, эмодзи, markdown
- Говори как живой человек по телефону
- Язык: русский
"""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def record_audio():
    """Record from microphone until Enter is pressed."""
    print("  [Говори... нажми Enter чтобы остановить]", flush=True)
    proc = subprocess.Popen(
        ["sox", "-d", "-r", "16000", "-c", "1", "-b", "16", WAV_IN],
        stderr=subprocess.PIPE,
    )
    input()  # wait for Enter
    proc.send_signal(signal.SIGINT)
    proc.wait()
    stderr = proc.stderr.read().decode() if proc.stderr else ""
    # Show recording info
    result = subprocess.run(["soxi", WAV_IN], capture_output=True, text=True)
    duration_line = [l for l in result.stdout.splitlines() if "Duration" in l]
    if duration_line:
        print(f"  📼 {duration_line[0].strip()}")
    if stderr and "WARN" in stderr:
        print(f"  ⚠ sox: {stderr.strip()}")


def transcribe():
    """Transcribe WAV file using whisper.cpp."""
    t0 = time.time()
    result = subprocess.run(
        [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", WAV_IN, "-l", "ru", "--no-timestamps"],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    text = result.stdout.strip()
    if result.returncode != 0:
        print(f"  ❌ whisper ошибка (код {result.returncode}): {result.stderr[:200]}")
        return ""
    # Clean up whisper artifacts
    text = re.sub(r"\[.*?\]", "", text).strip()
    print(f"  🎤 распознано ({elapsed:.1f}с): {text}")
    if not text:
        print(f"  whisper stdout: {result.stdout[:200]}")
        print(f"  whisper stderr: {result.stderr[:200]}")
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
                print(token, end="", flush=True)
                reply += token
            if chunk.get("done"):
                break

    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
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
    subprocess.Popen(["afplay", WAV_OUT])


def main():
    print("=== Балаболка (голосовой режим) ===")
    print(f"Модель: {MODEL}")
    print("Нажми Enter чтобы начать говорить, ещё раз Enter чтобы отправить.")
    print("Ctrl+C для выхода.")
    print()

    while True:
        try:
            input("▶ Нажми Enter и говори...")
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break

        record_audio()

        text = transcribe()
        if not text:
            print("  (ничего не распознано, попробуй ещё раз)\n")
            continue

        t0 = time.time()
        print("  Денис: ", end="", flush=True)
        try:
            reply, first_token_time = ask_ollama_stream(text)
        except Exception as e:
            print(f"Ошибка: {e}")
            continue

        t1 = time.time()
        ttft = first_token_time - t0 if first_token_time else 0
        total = t1 - t0
        print(f"\n  ⏱ ИИ: первый токен {ttft:.1f}с | всего {total:.1f}с")

        speak(reply)
        print()


if __name__ == "__main__":
    main()
