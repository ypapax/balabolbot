#!/usr/bin/env python3
import json
import subprocess
import time
import urllib.request

MODEL = "qwen3:8b"
VOICE_MODEL = subprocess.os.path.expanduser("~/piper-voices/ru_RU-denis-medium.onnx")
PIPER = subprocess.os.path.expanduser("~/piper-env/bin/piper")
WAV = "/tmp/balabolka_out.wav"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """мы разыграем диалог по телефону. Ты обычный человек, который живет своей обычной жизнью. Ты сейчас находишься дома.
 Это устный разговор, не чат.
Я к тебе обращаюсь каким-то именем и ты отвлекаешься на это имя, какое бы имя я тебе не назвал вначале. Значит ты запоминаешь это имя и общаешься с позиции этого человека.

Правила:
- Максимум 1-2 коротких предложения
- Никаких списков, нумерации, эмодзи, markdown
- Говори как живой человек по телефону
- Язык: русский
"""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]


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

    # Strip thinking tags if present
    import re
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

    messages.append({"role": "assistant", "content": reply})
    return reply, first_token_time


def speak(text):
    clean = text.replace("*", "").replace("#", "").replace("`", "")
    subprocess.run(
        [PIPER, "--model", VOICE_MODEL, "--output_file", WAV],
        input=clean.encode(),
        capture_output=True,
    )
    subprocess.Popen(["afplay", WAV])


def main():
    print("=== Балаболка ===")
    print(f"Модель: {MODEL}")
    print("Печатай сообщение и нажми Enter. Денис ответит голосом.")
    print("Для выхода: quit")
    print()

    while True:
        try:
            user_input = input("Ты: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        t0 = time.time()
        print("Денис: ", end="", flush=True)
        try:
            reply, first_token_time = ask_ollama_stream(user_input)
        except Exception as e:
            print(f"\rОшибка: {e}")
            continue

        t1 = time.time()
        ttft = first_token_time - t0 if first_token_time else 0
        total = t1 - t0
        print(f"\n  ⏱ первый токен: {ttft:.1f}с | всего: {total:.1f}с\n")

        speak(reply)


if __name__ == "__main__":
    main()
