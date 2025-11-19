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
├── build_vocabulary.py # Построение словаря
├── main.py # Основной скрипт для обучения
└── checkpoints/ # Папка для сохранения моделей
```
## Установка

1. Создайте виртуальное окружение:
```bash
python -m venv math2latex_env
source math2latex_env/bin/activate  # Linux/Mac
math2latex_env\Scripts\activate    # Windows
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Установите LaTeX и Ghostscript для корректной работы предпросмотра формул в GUI:

   **Для Windows:**
   - Установите [MiKTeX](https://miktex.org/download) (рекомендуется полная установка)
   - Установите [Ghostscript](https://www.ghostscript.com/download/gsdnld.html)
   - Убедитесь, что оба пакета добавлены в системную переменную PATH:
     - MiKTeX: обычно `C:\Program Files\MiKTeX\miktex\bin\x64\` или `C:\Users\<username>\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`
     - Ghostscript: обычно `C:\Program Files\gs\gs<version>\bin\`
   - Проверьте установку, выполнив в командной строке:
     ```bash
     latex --version
     dvipng --version
     gswin64c --version
     ```

   **Для Linux:**
   ```bash
   sudo apt-get install texlive-full ghostscript
   ```

   **Для macOS:**
   ```bash
   brew install --cask mactex
   brew install ghostscript
   ```

   После установки перезапустите приложение для применения изменений в PATH.

4. Подготовьте данные:
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
