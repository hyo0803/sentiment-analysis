# scripts/download_data.py
import argparse
from pathlib import Path
from src.data.download import download_kaggle_dataset
from src.config.utils import load_config

def main():
    cfg = load_config()
    default_dataset = cfg.get("dataset", {}).get("name")
    default_output = cfg.get("paths", {}).get("raw_data_dir", "data/raw")
    default_extract = cfg.get("dataset", {}).get("extract", True)

    parser = argparse.ArgumentParser(description="Загрузка датасета с Kaggle (конфиг: config/params.yaml)")
    parser.add_argument("--dataset", type=str, required=(default_dataset is None),
                        help=f"Название датасета 'user/dataset' или URL (по умолчанию из params.yaml: {default_dataset})")
    parser.add_argument("--output-dir", type=str, default=default_output,
                        help=f"Папка для сохранения данных (по умолчанию: {default_output})")
    parser.add_argument("--no-extract", action="store_true",
                        help="Отключить распаковку архива после скачивания (по умолчанию распаковывается, управляется флагом dataset.extract в params.yaml)")
    args = parser.parse_args()

    dataset = args.dataset or default_dataset
    if not dataset:
        parser.error("Название датасета обязательно либо через --dataset, либо в config/params.yaml (dataset.name).")
    
    output_dir = args.output_dir
    extract = (not args.no_extract) and bool(default_extract)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Скачивание датасета: {dataset} → {output_dir}  (extract={extract})")
    download_kaggle_dataset(dataset_name=dataset, output_dir=output_dir, extract=extract)

if __name__ == "__main__":
    main()