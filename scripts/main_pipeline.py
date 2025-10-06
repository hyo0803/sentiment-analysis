#!/usr/bin/env python3
"""
End-to-end pipeline for sentiment classification using SentenceTransformer + LinearSVC.
Steps:
1. Download dataset from Kaggle
2. Validate and normalize structure (ensure 'text' and 'label' columns)
3. Run EDA
4. Split into train/test
5. Train model on train
6. Evaluate on test → save metrics
7. Retrain final model on full dataset → save
"""

import os
import sys
import json
import pandas as pd 
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.setup import ensure_dependencies, ensure_kaggle_config
from src.data.download import download_kaggle_dataset  as download_dataset_from_kaggle
from src.data.utils import normalize_dataset as validate_and_normalize_data
from src.data.eda import generate_eda_report as run_eda
from src.models.classifier import SentimentClassifier


# === Конфигурация ===
DATASET_NAME = "utkarshx27/anime-recommendation-dataset"  # ← ЗАМЕНИ НА СВОЙ!
DATASET_FILE = "anime_recommendation_dataset.csv"
TEXT_COLUMN = "text"
TARGET_COLUMN = "label"  # должен содержать "positive"/"negative"

RAW_DATA_PATH = Path("data") / DATASET_FILE
PROCESSED_DATA_PATH = Path("data/raw/norm_anime_data.csv")
TEST_DATA_PATH = Path("data/test.csv")
REPORTS_DIR = Path("data/reports")
MODEL_DIR = Path("model_weights")
FINAL_MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"


def label_to_int(label):
    return 1 if label == "positive" else 0


def main():
    print("Проверка зависимостей и конфигурации...")
    ensure_dependencies()      # устанавливает kaggle, pandas и др., если нужно
    ensure_kaggle_config()     # копирует kaggle.json в ~/.kaggle/
    
    print("Запуск end-to-end пайплайна машинного обучения\n")
    # 1. Создаём директории
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Скачиваем датасет
    print("Шаг 1: Скачивание датасета с Kaggle...")
    download_dataset_from_kaggle(
        dataset_name=DATASET_NAME,
        output_path=RAW_DATA_PATH
    )

    # 3. Валидация и нормализация
    print("🧹 Шаг 2: Валидация и нормализация данных...")
    df = validate_and_normalize_data(
        input_path=RAW_DATA_PATH,
        text_column=TEXT_COLUMN,
        target_column=TARGET_COLUMN,
        output_path=PROCESSED_DATA_PATH
    )

    # Убеждаемся, что target — строки "positive"/"negative"
    assert df[TARGET_COLUMN].isin(["positive", "negative"]).all(), \
        "Целевая переменная должна содержать только 'positive' или 'negative'"

    # 4. EDA
    print("Шаг 3: Проведение EDA...")
    run_eda(df, output_dir=REPORTS_DIR)

    # 5. Разделение на train/test
    print("SplitOptions: 80/20, stratify по sentiment...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COLUMN]
    )
    test_df.to_csv(TEST_DATA_PATH, index=False)

    # 6. Подготовка данных для обучения
    X_train = train_df[TEXT_COLUMN].tolist()
    y_train_str = train_df[TARGET_COLUMN].tolist()
    y_train = [label_to_int(y) for y in y_train_str]

    X_test = test_df[TEXT_COLUMN].tolist()
    y_test_str = test_df[TARGET_COLUMN].tolist()
    y_test = [label_to_int(y) for y in y_test_str]

    # 7. Обучение модели на train
    print("🧠 Шаг 4: Обучение модели на тренировочных данных...")
    model = SentimentClassifier()
    model.fit(X_train, y_train)

    # 8. Предсказания на test
    print("📈 Шаг 5: Оценка на тестовых данных...")
    y_pred_str = model.predict(X_test)
    y_pred = [label_to_int(y) for y in y_pred_str]

    # 9. Расчёт метрик
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_pred))
    }
    print(f"Метрики на тесте: {json.dumps(metrics, indent=2)}")

    # 10. Сохранение метрик
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Метрики сохранены: {METRICS_PATH}")

    # 11. Обучение финальной модели на всех данных
    print("🔄 Шаг 6: Обучение финальной модели на полном датасете...")
    X_full = df[TEXT_COLUMN].tolist()
    y_full = [label_to_int(y) for y in df[TARGET_COLUMN].tolist()]
    final_model = SentimentClassifier()
    final_model.fit(X_full, y_full)

    # 12. Сохранение финальной модели
    final_model.save_model(FINAL_MODEL_PATH)
    print(f"✅ Финальная модель сохранена: {FINAL_MODEL_PATH}")

    print("\n🎉 Пайплайн успешно завершён!")


if __name__ == "__main__":
    main()