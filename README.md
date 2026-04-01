# Balabolka

Локальный голосовой бот на macOS — аналог голосового режима ChatGPT, но полностью бесплатный и оффлайн.

## Архитектура

```
Микрофон → [Whisper - распознавание речи] → текст
                                              ↓
                                     [Qwen3 8B - ответ ИИ]
                                              ↓
                                    [Piper/Denis - озвучка] → Динамик
```

## Требования к памяти (32GB RAM)

### Вариант 1: Qwen3:8b (рекомендуется)

| Компонент | RAM |
|-----------|-----|
| Qwen3:8b модель | ~5 GB |
| Контекст диалога + overhead | ~2 GB |
| Whisper (распознавание речи) | ~1-2 GB |
| Piper (озвучка) | ~0.3 GB |
| macOS + приложения | ~8-10 GB |
| **Итого** | **~17-19 GB** |
| **Свободно** | **~13-15 GB** |

Скорость ответа: ~30-40 токенов/сек — почти мгновенно.

### Вариант 2: Qwen3:14b (качественнее, но медленнее)

| Компонент | RAM |
|-----------|-----|
| Qwen3:14b модель | ~9 GB |
| Контекст диалога + overhead | ~2.5 GB |
| Whisper (распознавание речи) | ~1-2 GB |
| Piper (озвучка) | ~0.3 GB |
| macOS + приложения | ~8-10 GB |
| **Итого** | **~21-24 GB** |
| **Свободно** | **~8-11 GB** |

Скорость ответа: ~15-20 токенов/сек — задержка 1-2 сек перед ответом.

## Установка

### 1. Ollama + Qwen3

```bash
brew install ollama
brew services start ollama
ollama pull qwen3:8b
```

Тест:
```bash
ollama run qwen3:8b "Привет, как дела?"
```

### 2. Python venv + Piper TTS (озвучка)

```bash
python3 -m venv ~/piper-env
source ~/piper-env/bin/activate
pip install piper-tts pathvalidate
```

### 3. Скачать русский голос (Denis, мужской, бесплатный)

```bash
mkdir -p ~/piper-voices

curl -L -o ~/piper-voices/ru_RU-denis-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx"
curl -L -o ~/piper-voices/ru_RU-denis-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx.json"
```

Другие русские голоса: `dmitri`, `irina`, `ruslan` — заменить `denis` в URL.

### 4. Тест озвучки

```bash
source ~/piper-env/bin/activate
echo "Привет, я говорю по-русски" | piper \
  --model ~/piper-voices/ru_RU-denis-medium.onnx \
  --output_file /tmp/piper_out.wav && afplay /tmp/piper_out.wav
```

## Shell-функция (для быстрого вызова)

Добавить в `~/.zshrc` или `~/.bashrc`:

```bash
say_ru() {
  source ~/piper-env/bin/activate
  echo "$*" | piper --model ~/piper-voices/ru_RU-denis-medium.onnx --output_file /tmp/piper_out.wav
  afplay /tmp/piper_out.wav
}
```

Использование:

```bash
say_ru Привет, как дела?
```

## Все компоненты бесплатные

| Компонент | Лицензия | Работает оффлайн |
|-----------|----------|-------------------|
| Ollama | MIT | да |
| Qwen3 | Apache 2.0 | да |
| Piper TTS | MIT | да |
| Whisper | MIT | да |
