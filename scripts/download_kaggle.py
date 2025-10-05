# scripts/download_data.py
import argparse
from src.data.download import download_kaggle_dataset

def main():
    parser = argparse.ArgumentParser(description="Загрузка датасета с Kaggle")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Название датасета в формате 'user/dataset-name' или URL"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Папка для сохранения данных"
    )
    args = parser.parse_args()
    
    download_kaggle_dataset(args.dataset, args.output_dir)

if __name__ == "__main__":
    main()