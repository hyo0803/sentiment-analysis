# Модель по определению тональности текста

# 🎯 Sentiment Classification Pipeline

**End-to-end пайплайн машинного обучения для бинарной классификации тональности текста**  
на основе публичного датасета с [Kaggle](https://www.kaggle.com/).

> 📌 **Роль в проекте**: Data Scientist + ML/DevOps Engineer  
> 🔍 **Исследование и выбор модели**: выполнено Data Scientist в ноутбуках  
> ✅ **Финальная модель**: эмбеддер `SentenceTransformer` + классическая модель `LinearSVC` (выбрана по метрикам F1 и ROC-AUC)

---

## 📁 Структура проекта

```
sentiment-classification-model/
├── config/                 # Конфигурации (Kaggle API)
├── data/                   # Все данные
│   ├── raw/                # Исходный датасет
│   └── reports/            # Отчёты EDA и метрики
├── model_weights/          # Сохранённые модели
├── notebooks/              # Исследование (EDA, сравнение моделей)
├── scripts/                # Скрипты автоматизации
├── src/                    # Модульный код (data, models, utils)
├── tests/                  # Тесты
├── requirements.txt        # Зависимости
└── README.md
```

---

## 🚀 Быстрый старт

### 1. Клонируйте репозиторий
```bash
git clone https://github.com/ваш-username/sentiment-classification-model.git
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
python scripts/run_pipeline.py
```

Это выполнит:
- Скачивание датасета с Kaggle
- EDA → сохранение графиков в `data/reports/`
- Обучение финальной модели
- Оценку на тестовой выборке → метрики в `data/reports/metrics.json`
- Сохранение модели → `model_weights/sentiment_model.joblib`

---

## 🧪 Попробуйте модель

После запуска пайплайна вы можете протестировать модель в интерактивном режиме:

```bash
python scripts/predict.py
```

Пример:
```
Введите текст для анализа: I love this anime!
Предсказание: positive (уверенность: 0.94)
```

---
---

## 📊 Результаты исследования (от Data Scientist)

- **Датасет**: [Anime Recommendation Dataset](https://www.kaggle.com/datasets/utkarshx27/anime-recommendation-dataset) (~50k записей)
- **Модели, протестированные**: Logistic Regression, Random Forest, LinearSVC
- **Лучшая модель**: `LinearSVC` с эмбеддингами от `paraphrase-multilingual-MiniLM-L12-v2`
- **Метрики на тесте**:
  - Accuracy: 0.92
  - F1-score: 0.91
  - ROC-AUC: 0.96

> Подробности — в ноутбуках: `notebooks/research.ipynb`, `notebooks/baseline.ipynb`

---

## 🛠 Технологии и инструменты

- **ML**: `scikit-learn`, `sentence-transformers`
- **Автоматизация**: Python-скрипты (`scripts/`)
- **Качество кода**: `black`, `flake8` (запуск: `black . && flake8 .`)
- **Тестирование**: `pytest` (запуск: `python -m pytest`)

---

## 📄 Лицензия

Этот проект создан в учебных целях. Датасет распространяется под лицензией Kaggle.

---

> 💡 **Цель проекта**: продемонстрировать воспроизводимый, документированный и автоматизированный ML-пайплайн, соответствующий best practices DevOps в машинном обучении.