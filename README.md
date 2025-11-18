# math2latex

Проект для распознавания математических формул на изображениях и преобразования их в LaTeX-последовательности с использованием архитектуры Transformer.

## Архитектура проекта

- **Encoder**: ResNet50 для извлечения визуальных признаков
- **Decoder**: Transformer Decoder для генерации LaTeX-последовательности
- **Tokenizer**: Специализированный токенизатор для LaTeX
- **GUI**: PyQt5 интерфейс для удобного использования

## Структура проекта
```bash
├── dataset.py # Dataset и DataLoader'ы
├── model.py # Архитектура модели (Encoder-Decoder)
├── train.py # Процесс обучения
├── inference.py # Функции для инференса
├── interface.py # Графический интерфейс
├── vocab.py # Класс для работы с вокабуляром
├── latex_tokenizer.py # Токенизатор LaTeX
├── build_vocabulary.py# Построение словаря
├── main.py # Основной скрипт для обучения
└── checkpoints/ # Папка для сохранения моделей
```
## Установка

1. Создайте виртуальное окружение:
```bash
python -m venv math_ocr_env
source math_ocr_env/bin/activate  # Linux/Mac
math_ocr_env\Scripts\activate    # Windows
```

2. Установите зависимостей:
```bash
pip install -r requirements.txt
```

3. Подготовьте данные:
```bash
python build_vocabulary.py
```

## Структура проекта

**Обучение**
```bash
python build_vocabulary.py
```

**Запуск GUI**
```bash
python interface.py
```

**Тестирование инференса**
```bash
python inference.py
```

## Данные

Проект использует датасет hoang-quoc-trung/fusion-image-to-latex-datasets с Hugging Face Hub, содержащий пары "изображение-LaTeX".

## Особенности

- Поддержка beam search и greedy decoding
- Автоматическое определение устройста (CPU/GPU)
- Mixed precision training
- Визуализация результатов
