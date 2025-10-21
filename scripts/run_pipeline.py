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
6. Train best model (SentenceTransformer + LinearSVC)
7. Evaluate on test → compute and save metrics
8. Retrain final model on full dataset → save weights

Usage: python -m scripts.main_pipeline

#################################################################################
End-to-end пайплайн машинного обучения для бинарной классификации тональности.
Полностью автоматизированный, без жёстко заданных имён файлов.

Шаги:
1. Загрузка датасета с Kaggle
2. Автоматическое определение основного CSV файла
3. Нормализация данных → обеспечение наличия столбцов 'text' и 'label'
4. Проведение EDA → сохранение визуализаций и статистики
5. Разделение на train/test
6. Обучение лучшей модели (SentenceTransformer + LinearSVC)
7. Оценка на тестовой выборке → вычисление и сохранение метрик
8. Обучение финальной модели на полном датасете → сохранение весов

Использование: python -m scripts.main_pipeline
"""

import sys
import re
import pandas as pd # type: ignore
from pathlib import Path
from sklearn.model_selection import train_test_split # type: ignore
import numpy as np # type: ignore

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import your modules
from src.data.download import download_kaggle_dataset
from src.data.utils import normalize_dataset
from src.data.eda import generate_eda_report, compute_classification_metrics
from src.models.classifier import SentimentClassifier
from src.config.utils import load_config

# === CONFIGURATION (loaded from config/params.yaml via load_config) ===
cfg = load_config() or {}

# dataset
DATASET_NAME = cfg.get("dataset", {}).get("name", "https://www.kaggle.com/datasets/yaseminkh/ru-eng-sentiment-analysis")

# columns
ORIGINAL_TEXT_COLUMN = cfg.get("columns", {}).get("text", "text")
ORIGINAL_LABEL_COLUMN = cfg.get("columns", {}).get("label", "target")

# label mappings / cleaning
LABEL_MAPPING = cfg.get("label_mapping", {-1: 0, 1: 1})
LABEL_TEXT_MAPPING = cfg.get("label_text_mapping", {0: "negative", 1: "positive"})
CLEAR_EXTRA_CLASSES = cfg.get("clear_extra_classes", True)

# paths
paths_cfg = cfg.get("paths", {}) or {}
DATA_DIR = Path(paths_cfg.get("data_dir", "data"))
RAW_DATA_DIR = Path(paths_cfg.get("raw_data_dir", str(DATA_DIR / "raw")))
NORMALIZED_DATA_PATH = paths_cfg.get("normalized_data_path", "")
REPORTS_DIR = Path(paths_cfg.get("reports_dir", str(DATA_DIR / "reports")))
MODEL_DIR = Path(paths_cfg.get("model_dir", "models/weights"))
FINAL_MODEL_PATH = Path(paths_cfg.get("final_model_path", str(MODEL_DIR / "sentiment_model.joblib")))
METRICS_PATH = Path(paths_cfg.get("metrics_path", str(REPORTS_DIR / "metrics.json")))

# split params
SPLIT_CFG = cfg.get("split", {}) or {}
TEST_SIZE = float(SPLIT_CFG.get("test_size", 0.2))
RANDOM_STATE = int(SPLIT_CFG.get("random_state", 42))
STRATIFY = bool(SPLIT_CFG.get("stratify", True))


def main():
    print("| Запуск end-to-end пайплайна машинного обучения\n")

    # --- 1. Подготовка директорий ---
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("|- Шаг 1: Получение данных...")
    # === Новый рефакторинг выбора файла данных ===
    print("|-- Поиск нормализованного датасета / исходных данных...")

    allowed_exts = {".csv", ".xlsx"}
    norm_candidate = None
    raw_data_path = None

    # 1) Если явно указан NORMALIZED_DATA_PATH и файл существует — используем его
    if NORMALIZED_DATA_PATH:
        p = Path(NORMALIZED_DATA_PATH)
        if p.exists() and p.suffix.lower() in allowed_exts:
            norm_candidate = p
            print(f"|--- Используем указанный нормализованный файл: {norm_candidate}")
        elif p.exists():
            print(f"|--- Указанный файл найден, но формат '{p.suffix}' не поддерживается. Ожидается .csv или .xlsx")

    # 2) Ищем любые norm_*.csv/xlsx в RAW_DATA_DIR (вне зависимости от NORMALIZED_DATA_PATH)
    if norm_candidate is None:
        csv_matches = sorted(RAW_DATA_DIR.glob("norm_*.csv"))
        xlsx_matches = sorted(RAW_DATA_DIR.glob("norm_*.xlsx"))
        matches = csv_matches + xlsx_matches
        if matches:
            norm_candidate = matches[0]
            print(f"|--- Найден нормализованный датасет: {norm_candidate}")

    # Если нашли нормализованный файл — используем его как исходный для последующих шагов
    if norm_candidate is not None:
        raw_data_path = norm_candidate
        normalized_output_path = norm_candidate
        print(f"|--- Используем нормализованный файл: {raw_data_path}")
    else:
        # 3) Ищем необработанный файл датасета в RAW_DATA_DIR:
        base_name = str(DATASET_NAME).replace('https://www.kaggle.com/datasets/', '').split('/')[-1]
        derived_csv = RAW_DATA_DIR / (base_name.replace('-', '_') + '.csv')
        derived_xlsx = RAW_DATA_DIR / (base_name.replace('-', '_') + '.xlsx')

        if derived_csv.exists():
            raw_data_path = derived_csv
            print(f"|--- Найден необработанный CSV: {raw_data_path}")
        elif derived_xlsx.exists():
            raw_data_path = derived_xlsx
            print(f"|--- Найден необработанный XLSX: {raw_data_path}")
        else:
            # fallback: любой CSV/XLSX в папке raw (исключая norm_*)
            other_csv = next(RAW_DATA_DIR.glob("*.csv"), None)
            other_xlsx = next(RAW_DATA_DIR.glob("*.xlsx"), None)
            if other_csv and not other_csv.name.startswith("norm_"):
                raw_data_path = other_csv
                print(f"|--- Найден необработанный CSV (fallback): {raw_data_path}")
            elif other_xlsx and not other_xlsx.name.startswith("norm_"):
                raw_data_path = other_xlsx
                print(f"|--- Найден необработанный XLSX (fallback): {raw_data_path}")

        # 4) Если не найден — скачать с Kaggle
        if raw_data_path is None:
            print("|--- Данные не найдены локально — скачивание с Kaggle...")
            downloaded = download_kaggle_dataset(
                dataset_name=DATASET_NAME,
                output_dir=str(RAW_DATA_DIR),
                extract=True
            )
            raw_data_path = Path(downloaded)
            # Если скачан путь — если это папка/архив, попытаться найти CSV/XLSX внутри
            if raw_data_path.is_dir():
                # ищем первые подходящие файлы
                cand = next(raw_data_path.glob("*.csv"), None) or next(raw_data_path.glob("*.xlsx"), None)
                if cand:
                    raw_data_path = cand
            print(f"|--- Получен файл: {raw_data_path}")

        # 5) Нормализация: определяем, куда сохранять нормализованный файл
        normalized_output_path = Path(NORMALIZED_DATA_PATH) if NORMALIZED_DATA_PATH else (RAW_DATA_DIR / f"norm_{raw_data_path.name}")
        print(f"|-- Нормализация будет сохранена в: {normalized_output_path}")

        df = normalize_dataset(
            input_path=str(raw_data_path),
            output_path=str(normalized_output_path),
            text_column=ORIGINAL_TEXT_COLUMN,
            clear_extra_classes=CLEAR_EXTRA_CLASSES,
            label_column=ORIGINAL_LABEL_COLUMN,
            label_mapping=LABEL_MAPPING,
            input_sep=",",
            output_sep=","
        )
        # убедимся, что raw_data_path указывает теперь на нормализованный файл для дальнейших шагов
        raw_data_path = normalized_output_path

    # Если norm_candidate был найден ранее, всё ещё нужно загрузить DataFrame (если не загружен)
    if 'df' not in locals():
        # Загружаем нормализованный файл в df
        df = pd.read_csv(raw_data_path) if raw_data_path.suffix.lower() == ".csv" else pd.read_excel(raw_data_path)

    # --- 4-5. Загрузка нормализованных данных + EDA ---
    print("|- Шаг 3: Проведение EDA...")
    generate_eda_report(
        df, 
        report_dir=REPORTS_DIR,
        text_col=ORIGINAL_TEXT_COLUMN,
        label_col=ORIGINAL_LABEL_COLUMN,
        sep=','
    )
    
    # --- 6. Разделение на train/test ---
    print(f"|- Шаг 4: Разделение выборки (test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify={STRATIFY}...)")
    stratify_col = df[ORIGINAL_LABEL_COLUMN] if STRATIFY else None
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_col
    )
    TRAIN_DATA_PATH = DATA_DIR / Path("train"+ str(raw_data_path).split('/')[-1].split('.')[0]+'.csv')
    print(f"|-- Сохранение тренировочной выборки в: {TRAIN_DATA_PATH}")
    train_df.to_csv(TRAIN_DATA_PATH, index=False, encoding='utf-8')

    TEST_DATA_PATH = DATA_DIR / Path("test"+ str(raw_data_path).split('/')[-1].split('.')[0]+'.csv')
    print(f"|-- Сохранение тестовой выборки в: {TEST_DATA_PATH}")
    test_df.to_csv(TEST_DATA_PATH, index=False, encoding='utf-8')

    # --- 7. Подготовка данных ---
    X_train = train_df[ORIGINAL_TEXT_COLUMN].values
    y_train = train_df[ORIGINAL_LABEL_COLUMN].values

    X_test = test_df[ORIGINAL_TEXT_COLUMN].values
    y_test = test_df[ORIGINAL_LABEL_COLUMN].values

    # --- 8. Обучение модели ---
    print("|- Шаг 5: Обучение лучшей модели (SentenceTransformer + LinearSVC)...")
    model = SentimentClassifier()
    model.fit(X_train, y_train)

    # --- 9. Предсказания на тесте ---
    print("|- Шаг 6: Получение предсказаний на тестовой выборке...")
    y_pred = model.predict(X_test)

    # Получаем скор/вероятность для позитивного класса, если возможно
    try:
        y_proba = model.predict_proba(X_test)
        y_proba_arr = np.asarray(y_proba)
        if y_proba_arr.ndim == 2 and y_proba_arr.shape[1] >= 2:
            # предполагаем порядок [class0, class1]
            y_score = y_proba_arr[:, 1]
        else:
            # single-column probabilities (positive class) или flat array
            y_score = y_proba_arr.ravel()
    except Exception:
        try:
            y_score = model.decision_function(X_test)
            y_score = np.asarray(y_score).ravel()
        except Exception:
            # fallback: используем предсказанные метки как скор (плохо, но позволяет посчитать roc_auc при отсутствии probas)
            y_score = np.asarray(y_pred, dtype=float).ravel()

    # Конвертация предсказаний в числа для метрик
    y_pred = np.asarray(y_pred).astype(int).ravel().tolist()

    # --- 10. Расчёт метрик (выполняется и сохраняется в compute_classification_metrics) ---
    print("|- Шаг 7: Расчёт метрик...")
    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        report_dir=REPORTS_DIR,
        metrics_path=METRICS_PATH
    )
    print("|-- Метрики на тестовой выборке:")
    for name, value in metrics.items():
        print(f"|--  {name}: {value if value is not None else 'N/A'}")
    print(f"|-- Метрики сохранены: {METRICS_PATH}")

    # --- 11. Финальное обучение на всём датасете ---
    print("|- Шаг 8: Обучение финальной модели на полном датасете...")
    X_full = df[ORIGINAL_TEXT_COLUMN].values
    y_full = df[ORIGINAL_LABEL_COLUMN].values
    final_model = SentimentClassifier()
    final_model.fit(X_full, y_full)

    # --- 12. Сохранение модели (с версионированием, если файл уже есть) ---
    print("|- Шаг 9: Сохранение финальной модели...")
    final_model_path = FINAL_MODEL_PATH
    if final_model_path.exists():
        print("|-- Файл уже существует!")
        MODEL_NAME = FINAL_MODEL_PATH.stem
        MODEL_format = FINAL_MODEL_PATH.suffix.lstrip('.')
        # Поведение версионирования: добавляем _vN перед расширением
        m = re.search(r"(.*)_v(\d+)$", MODEL_NAME)
        if m:
            base = m.group(1)
            version = int(m.group(2)) + 1
        else:
            base = MODEL_NAME
            version = 2
        new_filename = f"{base}_v{version}.{MODEL_format}"
        final_model_path = final_model_path.parent / new_filename
        print(f"|-- Сохранение новой версии как: {final_model_path}")

    final_model.save_model(str(final_model_path))
    print(f"|-- Финальная модель сохранена: {final_model_path}")

    print("\n🎉 Пайплайн успешно завершён! Все результаты воспроизведены.")


if __name__ == "__main__":
    main()