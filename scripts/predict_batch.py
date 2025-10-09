# scripts/predict_batch.py
import argparse
from src.data.utils import load_data
from src.models.classifier import SentimentClassifier

def main():
    parser = argparse.ArgumentParser(description="Батч-инференс модели тональности")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Путь к сохранённой модели (.joblib)"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Путь к CSV/XLSX с колонкой 'text'"
    )
    parser.add_argument(
        "--input-data-sep",
        type=str,
        required=False,
        default=',',
        help="разделитель для CSV файла (например, ',' или ';')"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./data/predictions.csv",
        help="Путь для сохранения предсказаний"
    )
    
    args = parser.parse_args()
    
    # Загрузка данных
    df = load_data(args.input_path,
                   return_df=True,
                   sep=args.input_data_sep)

    # Инференс
    clf = SentimentClassifier(model_path=args.model_path)
    predictions = clf.predict(df["text"].tolist())
    predict_probas = clf.predict_proba(df["text"].tolist())
    
    # Сохранение
    df["prediction"] = predictions
    df["predict_proba"] = predict_probas
    
    if args.output_path.lower().endswith('.xlsx'):
        df.to_excel(args.output_path, index=False)
    elif args.output_path.lower().endswith('.csv'):
        df.to_csv(args.output_path, 
                  index=False,encoding='utf-8', 
                  sep=args.input_data_sep if args.input_data_sep else ',')
    print(f"Предсказания сохранены в {args.output_path}")

if __name__ == "__main__":
    main()