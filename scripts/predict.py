#!/usr/bin/env python3
# scripts/predict.py
import argparse
from src.models.classifier import SentimentClassifier
from src.data.utils import to_python_ints, parse_label_mapping
from src.config.utils import load_config
from pathlib import Path
import numpy as np # type: ignore
from src.data.eda import compute_classification_metrics

def _ensure_first(x):
    """Вернуть первый элемент из list/ndarray/скаляр в питон-типе."""
    val = to_python_ints(x)
    if isinstance(val, list):
        return val[0] if val else None
    return val

def _format_proba_for_pred(probas, pred):
    """
    Получить уверенность для предсказанного класса:
      - если probas — iterable of iterables, берём first_row и пытаемся взять элемент по индексу pred (int),
        иначе возвращаем max(first_row)
      - если probas — scalar, возвращаем его
    Возвращает float или None.
    """
    try:
        arr = np.asarray(probas)
        if arr.ndim == 0:
            return float(arr.item())
        # first row
        first = arr[0]
        # если это одномерный вектор вероятностей
        if hasattr(first, "__len__"):
            try:
                idx = int(pred)
            except Exception:
                idx = None
            if idx is not None and 0 <= idx < len(first):
                return float(first[idx])
            return float(np.max(first))
        else:
            return float(first)
    except Exception:
        return None

def _map_label_text(pred, mapping):
    """Безопасно сопоставляет предсказание в текст через mapping (поддерживает строковые ключи)."""
    if mapping is None:
        return str(pred)
    # попробуем использовать int ключ, затем строковый
    try:
        key = int(pred)
    except Exception:
        key = pred
    return mapping.get(key, mapping.get(str(key), str(pred)))

def main():
    parser = argparse.ArgumentParser(description="Инференс модели тональности")
    parser.add_argument("--model-path", type=str, required=False, help="Путь к сохранённой модели (.joblib)")
    parser.add_argument("--text", type=str, required=False, help="Текст для анализа тональности")
    parser.add_argument("--label-mapping", type=parse_label_mapping, required=False, help="Маппинг меток (JSON/Python literal)")
    parser.add_argument("--label-text-mapping", type=parse_label_mapping, required=False, help="Текстовая расшифровка меток")
    parser.add_argument("--true-label", type=str, required=False, help="(опционально) истинная метка для текста — для подсчёта метрик")
    args = parser.parse_args()

    # load config via shared loader
    cfg = load_config() or {}

    default_model = cfg.get("paths", {}).get("final_model_path", "./models/weights/sentiment_model.joblib")
    model_path = args.model_path or default_model
    label_mapping = args.label_mapping if args.label_mapping is not None else cfg.get("label_mapping", None)
    label_text_mapping = args.label_text_mapping if args.label_text_mapping is not None else cfg.get("label_text_mapping", {0: "negative", 1: "positive"})
    reports_dir = Path(cfg.get("paths", {}).get("reports_dir", "data/reports"))
    metrics_path = Path(cfg.get("paths", {}).get("metrics_path", reports_dir / "predict_metrics.json"))

    if not Path(model_path).exists():
        print(f"Warning: модель не найдена по пути {model_path}. Попробуйте указать корректный --model-path")

    model = SentimentClassifier(model_path=model_path)

    def infer_and_print(text):
        texts = [text]
        # predict с fallback'ами (разные сигнатуры)
        try:
            preds = model.predict(texts, label_mapping=label_mapping)
        except TypeError:
            try:
                preds = model.predict(texts, label_mapping)
            except TypeError:
                preds = model.predict(texts)
        pred = _ensure_first(preds)

        # попытка получить вероятности
        try:
            probas = model.predict_proba(texts)
        except Exception:
            probas = None

        conf = _format_proba_for_pred(probas, pred) if probas is not None else None
        pred_text = _map_label_text(pred, label_text_mapping)

        if conf is not None:
            print(f"Тональность текста: {pred_text} (класс предсказания: {pred}, proba: {conf:.4f})")
        else:
            print(f"Тональность текста: {pred_text} (proba: {pred})")

        # Если передана истинная метка — считаем и сохраняем метрики
        if args.true_label is not None:
            # Попытка привести true-label к числу через label_mapping, затем int
            true_val = args.true_label
            if label_mapping and true_val in label_mapping:
                true_val = label_mapping[true_val]
            else:
                try:
                    true_val = int(true_val)
                except Exception:
                    pass

            try:
                pred_val = int(pred)
            except Exception:
                pred_val = pred

            y_score_for_metrics = [conf] if conf is not None else None
            reports_dir.mkdir(parents=True, exist_ok=True)
            metrics = compute_classification_metrics(
                y_true=[true_val],
                y_pred=[pred_val],
                y_score=y_score_for_metrics,
                report_dir=reports_dir,
                metrics_path=metrics_path
            )
            print("Метрики для переданного примера:")
            for k, v in metrics.items():
                print(f"  {k}: {v if v is not None else 'N/A'}")

    if args.text:
        infer_and_print(args.text)
        return

    # интерактивный режим
    try:
        while True:
            text = input("\nВведите текст для анализа (или 'quit' для выхода): ")
            if text.strip().lower() in {"quit", "exit"}:
                break
            if not text.strip():
                print("Пустой ввод, попробуйте ещё.")
                continue
            infer_and_print(text)
    except (KeyboardInterrupt, EOFError):
        print("\nВыход.")

if __name__ == "__main__":
    main()