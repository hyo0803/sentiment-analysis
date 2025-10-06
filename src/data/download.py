# src/data/download.py
import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi  # явный импорт

def download_kaggle_dataset(
    dataset_name: str,
    output_dir: str = "data/raw",
    extract: bool = True
):
    """
    Скачивает датасет с Kaggle.
    
    Args:
        dataset_name: например, "user/dataset-name"
        output_dir: папка для сохранения
        extract: распаковывать ли архив
    """
    
    # Очистка URL (на случай копипасты)
    if "kaggle.com/datasets/" in dataset_name:
        dataset_name = dataset_name.split("kaggle.com/datasets/")[-1].strip()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Скачивание {dataset_name} в {output_dir}...")
    
    api = KaggleApi()
    api.authenticate()  # предполагается, что ~/.kaggle/kaggle.json уже настроен
    
    api.dataset_download_files(
        dataset_name,
        path=output_dir,
        unzip=False
    )
    
    if extract:
        zip_filename = f"{dataset_name.split('/')[-1]}.zip"
        zip_path = Path(output_dir) / zip_filename
        if zip_path.exists():
            print("Распаковка...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            zip_path.unlink()  # безопасное удаление через pathlib
    print("Загрузка завершена.")