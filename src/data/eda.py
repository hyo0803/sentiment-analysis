# src/data/eda.py
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path

from src.data.utils import load_data, install_package

def generate_eda_report(
    data_path: str,
    report_dir: str = "data/reports",
    text_col: str = "text",
    label_col: str = "label",
    sep=','
):
    """
    Генерирует простой EDA-отчёт: длина текстов, баланс классов.
    """
    install_package("seaborn")
    install_package("pandas")
    import seaborn as sns # type: ignore
    import pandas as pd # type: ignore
    
    df = load_data(data_path, return_df=True, sep=sep)
    
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Баланс классов
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=label_col)
    plt.title("Распределение классов")
    plt.savefig(f"{report_dir}/class_distribution.png")
    plt.close()
    
    # 2. Длина текстов
    df["text_len"] = df[text_col].astype(str).apply(len)
    plt.figure(figsize=(8, 5))
    sns.histplot(df, x="text_len", hue=label_col, bins=50)
    plt.title("Распределение длины текстов")
    plt.savefig(f"{report_dir}/text_length.png")
    plt.close()
    
    # 3. Статистика в CSV
    stats = {
        "dataset_shape": df.shape,
        "dataset_attributes": df.columns.tolist(),
        "total_samples": len(df),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "class_counts": df[label_col].value_counts().to_dict(),
        "attribute_nunique_values": df.nunique().to_dict(),
        "avg_text_length": df["text_len"].mean(),
    }
    pd.Series(stats).to_csv(f"{report_dir}/stats.csv", sep=',')
    
    print(f"EDA-отчёт сохранён в {report_dir}")