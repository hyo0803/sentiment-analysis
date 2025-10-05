# scripts/fit_model.py
import argparse
from pathlib import Path

from src.data.utils import load_data
from src.models.classifier import SentimentClassifier

def main():
    parser = argparse.ArgumentParser(description="Обучение модели тональности")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Путь к CSV/XLSX-файлу с данными (колонки: text, label)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/weights/sentiment_model.joblib",
        help="Путь для сохранения обученной модели"
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Модель эмбеддингов от sentence-transformers"
    )
    parser.add_argument( "--input-data-sep",
        type=str,
        required=False,
        help="разделитель для CSV файла (например, ',' или ';')")
    
    args = parser.parse_args()
    
    # Создаём папку для модели, если не существует
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Загрузка и обучение
    texts, labels = load_data(
        args.data_path,
        sep=args.input_data_sep if args.input_data_sep else ','
        )
    clf = SentimentClassifier(embedder=args.embedder)
    clf.fit(texts, labels)
    clf.save_model(args.model_path)

if __name__ == "__main__":
    main()