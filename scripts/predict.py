# scripts/predict.py
import argparse
from src.models.classifier import SentimentClassifier

def main():
    parser = argparse.ArgumentParser(description="Инференс модели тональности")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Путь к сохранённой модели (.joblib)"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Текст для анализа тональности"
    )
    
    args = parser.parse_args()
    
    clf = SentimentClassifier(model_path=args.model_path)
    prediction = clf.predict(args.text)
    predict_probas = clf.predict_proba(args.text)
    
    print(f"Текст: {args.text}")
    print(f"Тональность: {prediction}")
    print(f"Уверенность (predict_proba): {predict_probas:.2f}")

if __name__ == "__main__":
    main()