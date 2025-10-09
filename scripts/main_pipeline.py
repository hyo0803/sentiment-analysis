#!/usr/bin/env python3
"""
End-to-end ML pipeline for binary sentiment classification.
Fully automated, no hardcoded filenames.

Steps:
1. Download dataset from Kaggle
2. Automatically detect main CSV file
3. Normalize data → ensure 'text' and 'label' columns
4. Run EDA → save visualizations and stats
5. Split into train/test
6. Train best model (LinearSVC + SentenceTransformer)
7. Evaluate on test → compute and save metrics
8. Retrain final model on full dataset → save weights
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import your modules
from src.data.download import download_kaggle_dataset
from src.data.utils import normalize_dataset
from src.data.eda import generate_eda_report
from src.models.classifier import SentimentClassifier


# === CONFIGURATION ===
DATASET_NAME = "https://www.kaggle.com/datasets/yaseminkh/ru-eng-sentiment-analysis" 

# Колонки в исходном датасете (уточни по реальному CSV!)
ORIGINAL_TEXT_COLUMN = "text"      # ← например: "review", "comment", "text"
ORIGINAL_LABEL_COLUMN = "target"  # ← например: "rating", "label", "sentiment"

# Маппинг меток → 0/1 (обязательно!)
LABEL_MAPPING = {"negative":-1, "positive":1}

# Paths (без хардкода имён файлов!)
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
NORMALIZED_DATA_PATH = "" #Path("data/norm_dataset.csv")
# TEST_DATA_PATH = Path("data/test.csv")
REPORTS_DIR = DATA_DIR / "reports"
MODEL_DIR = Path("models/weights")
FINAL_MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"


def main():
    print("🚀 Запуск end-to-end пайплайна машинного обучения\n")

    # --- 1. Подготовка директорий ---
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataset_file_csv = RAW_DATA_DIR / Path(
        str(DATASET_NAME).replace('https://www.kaggle.com/datasets/', '').split('/')[-1].replace('-', '_') + '.csv'
        )
    dataset_file_xlsx = RAW_DATA_DIR / Path(
        str(DATASET_NAME).replace('https://www.kaggle.com/datasets/', '').split('/')[-1].replace('-', '_') + '.xlsx')

    
    print(dataset_file_csv, dataset_file_xlsx)
    if not os.path.exists(dataset_file_csv):
        if not os.path.exists(dataset_file_xlsx):
            # --- 2. Скачивание датасета + автоматическое определение CSV ---
            print("📥 Шаг 1: Скачивание датасета с Kaggle и поиск CSV-файла...")
            raw_data_path = download_kaggle_dataset(
                dataset_name=DATASET_NAME,
                output_dir=str(RAW_DATA_DIR),
                extract=True
            )
            print(f"✅ Используем файл: {raw_data_path}")
        else:
            raw_data_path = dataset_file_xlsx
    else:
        raw_data_path = dataset_file_csv
        
    # --- 3. Нормализация структуры ---
    print("🧹 Шаг 2: Нормализация данных (приведение к 'text' и 'label')...")
    df = normalize_dataset(
        input_path=str(raw_data_path),
        output_path=str(NORMALIZED_DATA_PATH) if NORMALIZED_DATA_PATH!="" else 'norm_'+str(raw_data_path),
        text_column=ORIGINAL_TEXT_COLUMN,
        label_column=ORIGINAL_LABEL_COLUMN,
        label_mapping=LABEL_MAPPING,
        input_sep=",",
        output_sep=","
    )

    # --- 4-5. Загрузка нормализованных данных + EDA ---
    print("📊 Шаг 3: Проведение EDA...")
    generate_eda_report(
        df, 
        report_dir=REPORTS_DIR,
        text_col= ORIGINAL_TEXT_COLUMN,
        label_col=ORIGINAL_LABEL_COLUMN,
        sep=','
    )
    
    # --- 6. Разделение на train/test ---
    print("SplitOptions: 80/20, стратификация по метке...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )
    
    TEST_DATA_PATH = DATA_DIR / Path("test"+ str(raw_data_path).split('/')[-1].split('.')[0]+'.csv')
    test_df.to_csv(TEST_DATA_PATH, index=False, encoding='utf-8')

    # --- 7. Подготовка данных ---
    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()

    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()

    # --- 8. Обучение модели ---
    print("🧠 Шаг 4: Обучение лучшей модели (LinearSVC + SentenceTransformer)...")
    model = SentimentClassifier()
    model.fit(X_train, y_train)

    # --- 9. Предсказания на тесте ---
    print("📈 Шаг 5: Получение предсказаний на тестовой выборке...")
    y_pred_str = model.predict(X_test, LABEL_MAPPING)

    # Конвертация в числа для метрик
    pred_to_int = LABEL_MAPPING#{v: k for k, v in LABEL_MAPPING.items()} #{"negative": 0, "positive": 1}
    y_pred = [pred_to_int[p] for p in y_pred_str]

    # --- 10. Расчёт метрик ---
    print("📊 Шаг 6: Расчёт метрик...")
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_pred))
    }

    print("Метрики на тестовой выборке:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # --- 11. Сохранение метрик ---
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Метрики сохранены: {METRICS_PATH}")

    # # --- 12. Финальное обучение на всём датасете ---
    # print("🔄 Шаг 7: Обучение финальной модели на полном датасете...")
    # X_full = df["text"].tolist()
    # y_full = df["label"].tolist()
    # final_model = SentimentClassifier()
    # final_model.fit(X_full, y_full)

    # --- 13. Сохранение модели ---
    FINAL_MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
    if FINAL_MODEL_PATH.exists():
        print("File already exists!")
        MODEL_PATH, MODEL_format = str(FINAL_MODEL_PATH).split('/')[-1].split('.')
        if MODEL_PATH[-1].isdigit():
            version = int(MODEL_PATH[-1]) + 1
        else:
            version = 2
        new_filename = MODEL_PATH[:-1] + str(version) + '.' + MODEL_format
        FINAL_MODEL_PATH = MODEL_DIR / new_filename
        print(f"Saving new version as: {FINAL_MODEL_PATH}")
 
    final_model = model
    final_model.save_model(str(FINAL_MODEL_PATH))
    print(f"✅ Финальная модель сохранена: {FINAL_MODEL_PATH}")

    print("\n🎉 Пайплайн успешно завершён! Все результаты воспроизведены.")


if __name__ == "__main__":
    main()