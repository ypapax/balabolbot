#!/bin/bash

MODEL="qwen3:8b"
VOICE_MODEL="$HOME/piper-voices/ru_RU-denis-medium.onnx"
PIPER="$HOME/piper-env/bin/piper"
WAV="/tmp/balabolka_out.wav"

SYSTEM_PROMPT="Ты голосовой ассистент Балаболка. Отвечай коротко, 1-3 предложения. Говори по-русски."

echo "=== Балаболка ==="
echo "Печатай сообщение и нажми Enter. Денис ответит голосом."
echo "Для выхода: quit"
echo ""

while true; do
    printf "Ты: "
    read -r input
    [ "$input" = "quit" ] && break
    [ -z "$input" ] && continue

    response=$(ollama run "$MODEL" --nowordwrap "$input" <<< "/set system $SYSTEM_PROMPT" 2>/dev/null)
    # fallback: simple call
    if [ -z "$response" ]; then
        response=$(echo "$input" | ollama run "$MODEL" 2>/dev/null)
    fi

    echo "Денис: $response"
    echo ""

    echo "$response" | "$PIPER" --model "$VOICE_MODEL" --output_file "$WAV" 2>/dev/null
    afplay "$WAV" 2>/dev/null &
done

echo "Пока!"
