# Sentiment Classification Model

Пайплайн для бинарной классификации тональности на базе SentenceTransformer + LinearSVC.  
Проект использует конфигурацию в `config/params.yaml` для всех путей и параметров.

Коротко:
- Код в `src/`
- Скрипты в `scripts/`
- Конфиг ― `config/params.yaml`
- Выходы: `data/`, `models/`, `data/reports/`

---

## 🚀 Быстрый старт

### 1. Клонируйте репозиторий
```bash
git clone https://github.com/hyo0803/sentiment-analysis.git
cd sentiment-classification-model
```

### 2. Установите зависимости
```bash
pip install -r requirements.txt
```

> ⚠️ **Требуется Python ≥3.8** 

### 3. Настройте Kaggle API
> 💡 Инструкция по получению `kaggle.json`: [Kaggle API Docs](https://www.kaggle.com/docs/api)
Поместите ваш файл `kaggle.json` (с API-ключом) в папку `config/`:
```bash
mkdir -p config
cp /путь/к/вашему/kaggle.json config/
```

### 4. Запустите полный пайплайн
```bash
python -m scripts.run_pipeline
```

Это выполнит:
- Скачивание датасета с Kaggle
- EDA → сохранение графиков в `data/reports/`
- Обучение финальной модели
- Оценку на тестовой выборке → метрики в `data/reports/metrics.json`
- Сохранение модели → `models/weights/sentiment_model.joblib`

---

## 🧪 Попробуйте модель

После запуска пайплайна вы можете протестировать модель в интерактивном режиме:

```bash
python -m scripts.predict --text "ТЕКСТ"
```

Пример:
```
Введите текст для анализа вместо "ТЕКСТ": "I love this anime!"
Предсказание: positive (уверенность: 0.94)
```

---

## 📁 Структура проекта

```
sentiment-classification-model/
├── config/                 # Конфигурации (Kaggle API, params.yaml)
├── data/                   # Все данные
│   ├── raw/                # Исходный датасет
│   ├── embeddings/         # Кэш эмбеддингов
│   └── reports/            # Отчёты EDA и метрики
├── models
│   └── weights/            # Сохранённые модели
├── notebooks/              # Исследование (EDA, сравнение моделей)
├── scripts/                # Скрипты автоматизации (CLI)
├── src/                    # Модульный код (data, models, utils)
├── tests/                  # Тесты
├── requirements.txt        # Зависимости
└── README.md
```

---

## 📄 Лицензия

Этот проект создан в учебных целях. Датасет распространяется под лицензией Kaggle.

---

> 💡 **Цель проекта**: продемонстрировать воспроизводимый, документированный и автоматизированный ML-пайплайн, соответствующий best practices DevOps в машинном обучении.

---

## Требования

- Python 3.8+
- Рекомендуется использовать virtualenv

---

## Установка (рекомендовано)

```bash
python -m venv .venv
source .venv/bin/activate  # или .venv\Scripts\activate на Windows
pip install --upgrade pip
pip install -r requirements.txt         # если есть
pip install -r requirements-dev.txt     # если есть dev-зависимости (pytest и т.п.)
```

---

## Конфигурация

- Основные параметры (пути, колонки, split, label mapping и пр.) задаются в `config/params.yaml`.
- CLI-аргументы в скриптах переопределяют значения из конфига.

---

## Основные скрипты и примеры использования

- Скачать датасет с Kaggle (dataset: user/dataset-name или полный URL)
  ```bash
  python -m scripts.download_kaggle --dataset "user/dataset-name" --output-dir "data/raw"
  ```
  - По умолчанию берётся `dataset.name` из `config/params.yaml`.

- Нормализация данных (получение колонок `text` и `label`)
  ```bash
  python -m scripts.normalize_data_struct --input-path "data/raw/file.csv"
  ```
  - CLI: --text-column, --label-column, --label-mapping (JSON / Python literal), --output-path

- Генерация EDA
  ```bash
  python -m scripts.run_eda --input-path "data/norm_file.csv"
  ```
  - Опции: --text-column, --label-column, --report-dir

- Обучение модели (fit)
  ```bash
  python -m scripts.fit_model --data-path "data/norm_file.csv" --model-path "models/weights/sentiment_model.joblib"
  ```
  - Можно задать embedder, cache-dir и т.п. через args или config.

- Интерактивный/одиночный инференс
  ```bash
  python -m scripts.predict --text "Это отличный продукт" --model-path "models/weights/sentiment_model.joblib"
  ```
  - Опция --true-label для расчёта и сохранения метрик для одного примера

- Батч-инференс (CSV/XLSX)
  ```bash
  python -m scripts.predict_batch --input-path "data/to_predict.csv" --output-path "data/predictions.csv"
  ```

- Полный пайплайн (скачать → нормализовать → EDA → train → eval → финальная модель)
  ```bash
  python -m scripts.main_pipeline
  ```
  - Параметры пайплайна читаются из `config/params.yaml`. CLI-аргументы в отдельных шагах остаются доступными.

---

## Тесты

- Используется pytest.
- Запуск (из корня проекта):
  ```bash
  PYTHONPATH=. pytest -q
  ```
  - или: `python -m pytest -q`
- Местоположение тестов: `tests/`

---

## Файловая организация (основные)

```
src/
  ├── data/ (utils, eda, download)
  ├── models/ (classifier.py)
  └── config/ (utils.py)
scripts/ (CLI-утилиты)
config/params.yaml — основной конфиг
```

---

## Полезные советы

- Если импорты не находятся при запуске pytest — убедитесь, что запускаете из корня репозитория и PYTHONPATH=. или установите пакет в editable режиме:
  ```bash
  pip install -e .
  ```

---

## Поддержка и вклад

- Для изменений конфигурации правьте `config/params.yaml`.
- Добавляйте тесты в `tests/` при изменении логики.