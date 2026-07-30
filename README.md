# 🧠 CAPTCHA Solver — AI-powered CAPTCHA solving

AI-модель для решения CAPTCHA (hCaptcha, reCAPTCHA v2, текстовые) на основе CLIP zero-shot классификации.

## Как работает

### Image CAPTCHA (hCaptcha / reCAPTCHA v2)
1. Получает grid-изображение (3x3 или 4x4) или отдельные тайлы
2. Разрезает на отдельные тайлы
3. CLIP вычисляет similarity каждого тайла с positive/negative промптами
4. Возвращает индексы тайлов, где изображён нужный объект

### Text CAPTCHA
1. Preprocessing: бинаризация, деноизинг, контрастность
2. EasyOCR распознаёт текст
3. Возвращает строку с confidence

## Быстрый старт

### 1. Установка (на сервере с GPU)

```bash
# Создай виртуальное окружение
python -m venv venv
source venv/bin/activate

# Установи зависимости
pip install -r requirements.txt
```

### 2. Запуск API сервера

```bash
# Из корня проекта:
python -m server.app

# Или через uvicorn:
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Сервер запустится на `http://0.0.0.0:8000`. Документация API: `http://localhost:8000/docs`

### 3. Использование API

```python
import httpx

# Решить image CAPTCHA (grid)
response = httpx.post("http://YOUR_SERVER:8000/solve/image", json={
    "image_url": "https://example.com/captcha.png",
    "prompt": "Select all images with a bus",
    "grid": "3x3",
})
print(response.json())
# {"selected_tiles": [0, 3, 6], "confidence": [0.85, 0.72, 0.91], ...}

# Решить image CAPTCHA (отдельные тайлы — как в hCaptcha)
response = httpx.post("http://YOUR_SERVER:8000/solve/image", json={
    "tile_urls": ["url1", "url2", ..., "url9"],
    "prompt": "Please click each image containing a bus",
})

# Решить text CAPTCHA
response = httpx.post("http://YOUR_SERVER:8000/solve/text", json={
    "image_url": "https://example.com/text_captcha.png",
})
print(response.json())
# {"text": "abc123", "confidence": 0.95}

# Классифицировать изображение (для отладки)
response = httpx.post("http://YOUR_SERVER:8000/classify", json={
    "image_url": "https://example.com/image.png",
})
```

### 4. Использование как библиотеки

```python
from src.clip_solver import CLIPSolver
from src.ocr_solver import OCRSolver

# Image CAPTCHA
solver = CLIPSolver(model_name="ViT-B/32", threshold=0.5)
result = solver.solve(
    image_source="captcha.png",
    prompt="Select all images with a bus",
    grid="3x3",
)
print(result.selected_indices)  # [0, 3, 6]

# Text CAPTCHA
ocr = OCRSolver(languages=["en"])
result = ocr.solve("text_captcha.png")
print(result.text)  # "abc123"
```

## API Endpoints

| Method | Endpoint | Описание |
|--------|----------|----------|
| `POST` | `/solve/image` | Решить image grid CAPTCHA |
| `POST` | `/solve/text` | Решить text CAPTCHA (OCR) |
| `POST` | `/classify` | Классифицировать изображение |
| `GET` | `/health` | Health check |
| `GET` | `/categories` | Список категорий |

## Конфигурация

Настройки в `config.yaml`:

```yaml
clip:
  model: "ViT-B/32"   # Модель CLIP
  threshold: 0.50      # Порог confidence
ocr:
  languages: ["en"]
  gpu: true
server:
  host: "0.0.0.0"
  port: 8000
```

### Модели CLIP

| Модель | Скорость | Точность | VRAM |
|--------|----------|----------|------|
| `ViT-B/32` | ⚡ Быстрая | Хорошая | ~1 GB |
| `ViT-B/16` | ⚡ Средняя | Лучше | ~1.5 GB |
| `ViT-L/14` | 🐢 Медленная | Лучшая | ~3 GB |
| `ViT-L/14@336px` | 🐢 Самая медленная | Максимальная | ~4 GB |

## Тесты

```bash
# Тесты (без GPU — мокаются)
python -m pytest tests/ -v
```

## Структура

```
captcha-model/
├── src/
│   ├── clip_solver.py    # CLIP image solver
│   ├── ocr_solver.py     # Text OCR solver
│   ├── image_utils.py    # Image processing
│   ├── prompts.py        # CLIP prompt templates
│   └── captcha_types.py  # Data models
├── server/
│   └── app.py            # FastAPI API server
├── examples/
│   └── demo.py           # Demo script
├── tests/
│   └── test_solver.py    # Unit tests
├── config.yaml           # Configuration
└── requirements.txt      # Dependencies
```
# anchous
# anchous
# anchous
