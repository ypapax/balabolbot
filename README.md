# Balabolka

Локальный голосовой бот на macOS — аналог голосового режима ChatGPT, но полностью бесплатный и оффлайн.

## Архитектура

```
Микрофон → [Whisper - распознавание речи] → текст
                                              ↓
                                     [Qwen2.5 7B - ответ ИИ]
                                              ↓
                                    [Piper/Denis - озвучка] → Динамик
```

## Бенчмарки (MacBook M3, 32GB RAM)

| Модель | Первый токен | Полный ответ (1-2 предложения) | RAM | Примечание |
|--------|-------------|-------------------------------|-----|------------|
| **qwen2.5:7b** | **0.2с** | **0.5-1.1с** | **4.8 GB** | **без thinking, рекомендуется** |
| qwen3:8b | 3.4-5.0с | 3.8-7.2с | 5.2 GB | thinking тратит ~150 скрытых токенов |
| qwen3:4b | 15-40с | 16-42с | 3.2 GB | thinking тратит ~230 скрытых токенов |

Qwen3 серия использует "thinking" — модель генерирует скрытые токены размышлений перед ответом.
Для голосового бота это критично: задержка 3-40с вместо 0.2с. Qwen2.5 не имеет thinking и отвечает мгновенно.

Генерация голоса (Piper): ~0.7с — не является узким местом.

## Установка

### 1. Ollama + Qwen2.5

```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5:7b
```

Тест:
```bash
ollama run qwen2.5:7b "Привет, как дела?"
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

## Запуск бота

```bash
source ~/piper-env/bin/activate
python3 ~/code/voice_bot/chat.py
```

Или с другой моделью:
```bash
python3 ~/code/voice_bot/chat.py qwen3:8b
```

## Все компоненты бесплатные

| Компонент | Лицензия | Работает оффлайн |
|-----------|----------|-------------------|
| Ollama | MIT | да |
| Qwen2.5 | Apache 2.0 | да |
| Piper TTS | MIT | да |
| Whisper | MIT | да |
