# src/data/eda.py
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import json
from pathlib import Path
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
from typing import Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
) # type: ignore

from src.data.utils import load_data
from src.config.utils import load_config

def generate_eda_report(
    data: Union[str, pd.DataFrame],
    report_dir: str = None,
    text_col: str = "text",
    label_col: str = "label",
    sep=','
):
    """
    Генерирует EDA-отчёт: сохраняет графики и статистику.
    Возвращает dict с путями к сохранённым артефактам.
    """
    # report_dir default from config if not provided
    if report_dir is None:
        cfg = load_config() or {}
        report_dir = cfg.get("paths", {}).get("reports_dir", "data/reports")
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load data if path given
    if isinstance(data, str):
        df = load_data(data, return_df=True, sep=sep, check_text_column=False)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()  # avoid mutating caller's df
    else:
        raise ValueError("data must be a file path or a pandas DataFrame")

    # Validate required columns
    if text_col not in df.columns:
        raise ValueError(f"text_col '{text_col}' not found in dataframe. Available columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in dataframe. Available columns: {list(df.columns)}")

    artifacts = {}

    # 1. Class distribution
    try:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=label_col, order=sorted(df[label_col].dropna().unique()))
        plt.title("Распределение классов")
        class_plot_path = report_dir / "class_distribution.png"
        plt.savefig(class_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        artifacts["class_distribution"] = str(class_plot_path)
    except Exception as e:
        artifacts["class_distribution_error"] = str(e)

    # 2. Text length distribution
    try:
        df["_eda_text_len"] = df[text_col].astype(str).map(lambda x: len(x))
        plt.figure(figsize=(8, 5))
        sns.histplot(df, x="_eda_text_len", hue=label_col, bins=50, element="step", stat="count")
        plt.title("Распределение длины текстов")
        textlen_plot_path = report_dir / "text_length.png"
        plt.savefig(textlen_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        artifacts["text_length"] = str(textlen_plot_path)
    except Exception as e:
        artifacts["text_length_error"] = str(e)

    # 3. Stats (JSON + CSV)
    try:
        stats = {
            "dataset_shape": list(df.shape),
            "dataset_attributes": df.columns.tolist(),
            "total_samples": int(len(df)),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            "missing_values": {col: int(cnt) for col, cnt in df.isnull().sum().to_dict().items()},
            "class_counts": {str(k): int(v) for k, v in df[label_col].value_counts().to_dict().items()},
            "attribute_nunique_values": {col: int(n) for col, n in df.nunique().to_dict().items()},
            "avg_text_length": float(df["_eda_text_len"].mean()) if "_eda_text_len" in df.columns else None,
        }
        stats_json_path = report_dir / "stats.json"
        with open(stats_json_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2)
        artifacts["stats_json"] = str(stats_json_path)

        # Also save CSV summary (one-line friendly)
        stats_series = pd.Series({k: (v if isinstance(v, (str, int, float, list, dict)) else str(v)) for k, v in stats.items()})
        stats_csv_path = report_dir / "stats.csv"
        stats_series.to_csv(stats_csv_path, sep=',')
        artifacts["stats_csv"] = str(stats_csv_path)
    except Exception as e:
        artifacts["stats_error"] = str(e)

    # Cleanup helper column
    if "_eda_text_len" in df.columns:
        try:
            df.drop(columns=["_eda_text_len"], inplace=True)
        except Exception:
            pass

    print(f"EDA-отчёт сохранён в {report_dir}")
    return artifacts

def compute_classification_metrics(
    y_true,
    y_pred,
    y_score=None,
    report_dir: Union[str, Path] = "data/reports",
    metrics_path: Union[str, Path, None] = None,
    pos_label: int = 1
):
    """
    Вычисляет метрики бинарной классификации и (если возможно) строит ROC-кривую.
    Сохраняет metrics.json в metrics_path (по умолчанию report_dir/metrics.json)
    и roc_curve.png в report_dir.
    Возвращает словарь метрик.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    if metrics_path is None:
        metrics_path = report_dir / "metrics.json"
    else:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
    metrics["precision"] = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    metrics["recall"] = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    metrics["f1"] = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    roc_auc_val = None
    if y_score is not None:
        try:
            y_score_arr = np.asarray(y_score).ravel()
            roc_auc_val = float(roc_auc_score(y_true_arr, y_score_arr))
            metrics["roc_auc"] = roc_auc_val
        except Exception:
            metrics["roc_auc"] = None
    else:
        try:
            y_score_arr = y_pred_arr.astype(float)
            roc_auc_val = float(roc_auc_score(y_true_arr, y_score_arr))
            metrics["roc_auc"] = roc_auc_val
        except Exception:
            metrics["roc_auc"] = None

    # Сохранение метрик
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=4)

    # Построение ROC, если есть y_score
    try:
        if metrics.get("roc_auc") is not None:
            fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
            roc_auc_val_ = auc(fpr, tpr)
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc_val_:.4f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic")
            plt.legend(loc="lower right")
            roc_path = report_dir / "roc_curve.png"
            plt.savefig(roc_path, dpi=300, bbox_inches="tight")
            plt.close()
    except Exception:
        pass

    return metrics