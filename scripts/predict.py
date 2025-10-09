# scripts/predict.py
import argparse
from src.models.classifier import SentimentClassifier

def main():
    parser = argparse.ArgumentParser(description="Инференс модели тональности")
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="Путь к сохранённой модели (.joblib)"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        help="Текст для анализа тональности"
    )
    
    args = parser.parse_args()
    
    model_path = args.model_path if args.model_path else "src/models/weights/sentiment_model.joblib"
    model = SentimentClassifier(model_path=model_path)
    
    if args.text:
        pred = model.predict(args.text)
        proba = model.predict_proba(args.text)
        print(f"Тональность текста: {pred} (уверенность: {proba:.2f})")
    else:
        while True:
            text = input("\nВведите текст для анализа (или 'quit' для выхода): ")
            if text.lower() == "quit":
                break
            pred = model.predict(text)
            proba = model.predict_proba(text)
            print(f"Тональность текста: {pred} (уверенность: {proba:.2f})")
            

if __name__ == "__main__":
    main()