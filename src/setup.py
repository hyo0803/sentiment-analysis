# src/setup.py
import os
import sys
import subprocess
import shutil
from pathlib import Path

def _is_package_installed(package: str) -> bool:
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def ensure_dependencies():
    """Устанавливает обязательные пакеты, если они не установлены."""
    required = ["kaggle", "pandas", "scikit-learn", "sentence-transformers", "seaborn"]
    for pkg in required:
        if not _is_package_installed(pkg):
            print(f"Установка {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def ensure_kaggle_config():
    """Копирует kaggle.json из config/ в ~/.kaggle/ с правами 600."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    src = Path("config/kaggle.json")
    dst = kaggle_dir / "kaggle.json"
    
    if not dst.exists():
        if not src.exists():
            raise FileNotFoundError(
                "Файл config/kaggle.json не найден. "
                "Поместите туда ваш API-ключ Kaggle."
            )
        shutil.copy(src, dst)
        os.chmod(dst, 0o600)
        print("Конфигурация Kaggle настроена.")
    