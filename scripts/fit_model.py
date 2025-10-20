#!/usr/bin/env python3
# scripts/fit_model.py
import argparse
from pathlib import Path

from src.data.utils import load_data
from src.models.classifier import SentimentClassifier
from src.config.utils import load_config

def main():
    # загрузка конфигурации проекта (config/params.yaml)
    cfg = load_config() or {}

    # значения по умолчанию из конфига
    paths_cfg = cfg.get("paths", {}) or {}
    training_cfg = cfg.get("training", {}) or {}
    columns_cfg = cfg.get("columns", {}) or {}

    default_data = paths_cfg.get("normalized_data_path") or paths_cfg.get("raw_data_dir") or None
    default_model_path = paths_cfg.get("final_model_path", "models/weights/sentiment_model.joblib")
    default_embedder = training_cfg.get("embedder", training_cfg.get("model_embedder", "paraphrase-multilingual-MiniLM-L12-v2"))
    default_cache = paths_cfg.get("embeddings_dir", "data/embeddings")
    default_sep = ","
    default_text_col = columns_cfg.get("text", "text")
    default_label_col = columns_cfg.get("label", "label")

    parser = argparse.ArgumentParser(description="Обучение модели тональности")
    parser.add_argument("--data-path", type=str, required=False,
                        help=f"Путь к CSV/XLSX с данными (колонки: {default_text_col}, {default_label_col}). По умолчанию берётся из config/params.yaml")
    parser.add_argument("--model-path", type=str, default=None,
                        help=f"Путь для сохранения обученной модели (по умолчанию: {default_model_path})")
    parser.add_argument("--embedder", type=str, default=None,
                        help=f"SentenceTransformer embedder (по умолчанию: {default_embedder})")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help=f"Папка для кэша эмбеддингов (по умолчанию: {default_cache})")
    parser.add_argument("--input-data-sep", type=str, required=False, default=None,
                        help="разделитель для CSV файла (например, ',' или ';')")

    args = parser.parse_args()

    # resolve effective params: CLI overrides config
    data_path = args.data_path or default_data
    if data_path is None:
        parser.error("Путь к данным не указан. Передайте --data-path или задайте paths.normalized_data_path в config/params.yaml")

    model_path = Path(args.model_path or default_model_path)
    embedder = args.embedder or default_embedder
    cache_dir = args.cache_dir or default_cache
    sep = args.input_data_sep or default_sep

    # ensure model dir exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"Загрузка данных из: {data_path} (sep='{sep}')")
    texts, labels = load_data(str(data_path), sep=sep)

    print(f"Инициализация модели эмбеддингов: {embedder}")
    clf = SentimentClassifier(embedder=embedder)

    print("Запуск обучения...")
    clf.fit(texts, labels, cache_dir=cache_dir)

    print(f"Сохранение модели в: {model_path}")
    clf.save_model(str(model_path))

    print("Обучение завершено успешно.")

if __name__ == "__main__":
    main()