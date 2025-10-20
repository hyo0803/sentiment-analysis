# src/data/download.py
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi  # явный импорт
import glob
from typing import Optional, List

def _normalize_dataset_name(dataset_name: str) -> str:
    """Привести dataset_name к форме user/dataset-name."""
    if "kaggle.com/datasets/" in dataset_name:
        dataset_name = dataset_name.split("kaggle.com/datasets/")[-1].strip()
    return dataset_name

def _find_candidate_file(output_dir: Path, slug: str) -> Optional[Path]:
    """
    Ищет подходящий CSV/XLSX в output_dir или в подпапке slug.
    Возвращает Path первого найденного файла или None.
    """
    patterns = [
        f"{slug}*.csv", f"{slug}*.xlsx",  # файлы, начинающиеся с slug
        "*.csv", "*.xlsx"  # любой CSV/XLSX в папке
    ]
    # сначала проверяем подпапку с именем slug
    subfolder = output_dir / slug
    search_dirs: List[Path] = [output_dir]
    if subfolder.exists() and subfolder.is_dir():
        search_dirs.insert(0, subfolder)

    for d in search_dirs:
        for pat in patterns:
            matches = sorted(d.glob(pat))
            if matches:
                return matches[0]
    return None

def download_kaggle_dataset(
    dataset_name: str,
    output_dir: str = "data/raw",
    extract: bool = True
) -> Path:
    """
    Скачивает датасет с Kaggle и возвращает путь к основному файлу (CSV/XLSX) или
    бросает FileNotFoundError.

    Args:
        dataset_name: например, "user/dataset-name" или полный URL
        output_dir: папка для сохранения
        extract: распаковывать ли архив

    Returns:
        Path к найденному CSV/XLSX
    """
    dataset_name = _normalize_dataset_name(dataset_name)
    slug = dataset_name.split('/')[-1]

    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    print(f"Скачивание {dataset_name} в {outp}...")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        raise RuntimeError("Kaggle API authentication failed. Ensure ~/.kaggle/kaggle.json is configured.") from e

    try:
        api.dataset_download_files(
            dataset_name,
            path=str(outp),
            unzip=False
        )
    except Exception as e:
        raise RuntimeError(f"Ошибка при скачивании датасета {dataset_name}: {e}") from e

    extracted_files: List[str] = []
    # Попытка распаковки zip (если просили extract)
    if extract:
        zip_filename = f"{slug}.zip"
        zip_path = outp / zip_filename
        if zip_path.exists():
            print("Распаковка архива...")
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(path=str(outp))
                    extracted_files = zf.namelist()
                # удаляем архив только после успешной распаковки
                try:
                    zip_path.unlink()
                except Exception:
                    print(f"Warning: не удалось удалить архив {zip_path}")
                print(f"Извлечено {len(extracted_files)} файлов.")
            except zipfile.BadZipFile:
                print(f"Внимание: {zip_path} не является корректным zip-архивом.")
            except Exception as e:
                print(f"Ошибка при распаковке архива: {e}")

    # Ищем подходящий файл среди извлечённых/в папке
    candidate = _find_candidate_file(outp, slug)
    if candidate:
        print(f"Найден файл: {candidate}")
        return candidate

    # Если ничего не найдено среди извлечённых, попытаться вернуть какой-либо подходящий файл
    all_matches = list(outp.glob("*.csv")) + list(outp.glob("*.xlsx"))
    if all_matches:
        # возвращаем первый попавшийся
        print(f"Найден подходящий файл: {all_matches[0]}")
        return all_matches[0]

    # Ничего не найдено — информируем пользователя
    raise FileNotFoundError(f"Не удалось найти CSV/XLSX для датасета {dataset_name} в {outp}. "
                            "Проверьте содержимое папки и корректность dataset_name.")