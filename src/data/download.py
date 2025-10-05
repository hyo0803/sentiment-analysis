# src/data/download.py
import os
import zipfile
from pathlib import Path

from src.data.utils import install_package, setup_kaggle_config

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
    # Установка необходимых пакетов
    install_package("kaggle")
    install_package("pandas")
    # Настройка конфигурации Kaggle
    setup_kaggle_config()

    import kaggle # type: ignore
    kaggle.api.authenticate()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dataset_name = dataset_name.replace('https://www.kaggle.com/datasets/', '')
    
    print(f"Скачивание {dataset_name} в {output_dir}...")
    kaggle.api.dataset_download_files(
      dataset_name,
      path=output_dir,
      unzip=False)
    
    if extract:
        zip_path = os.path.join(output_dir, f"{dataset_name.split('/')[-1]}.zip")
        if os.path.exists(zip_path):
            print("Распаковка...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(zip_path)  # удалить архив
    print("Загрузка завершена.")