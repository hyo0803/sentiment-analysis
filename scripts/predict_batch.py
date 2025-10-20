#!/usr/bin/env python3
# scripts/predict_batch.py
import argparse
import json
from pathlib import Path
from src.data.utils import load_data, to_python_ints, parse_label_mapping
from src.models.classifier import SentimentClassifier
from src.config.utils import load_config
import numpy as np # type: ignore

def _ensure_list_for_df(x, length):
    """Гарантирует, что x представлен как список длины length (распространяет скаляр при необходимости)."""
    if x is None:
        return [None] * length
    # numpy scalar or python scalar
    if isinstance(x, (np.generic, int, float, str)):
        return [to_python_ints(x)] * length
    try:
        lst = list(x)
    except Exception:
        return [x] * length
    if len(lst) == length:
        return lst
    if len(lst) == 1:
        return lst * length
    return [to_python_ints(el) for el in lst]

def _map_pred_to_text(pred, mapping):
    """Безопасная текстовая расшифровка метки через mapping."""
    if mapping is None:
        return str(pred)
    try:
        key = int(pred)
    except Exception:
        key = pred
    return mapping.get(key, mapping.get(str(key), str(pred)))

def main():
    parser = argparse.ArgumentParser(description="Батч-инференс модели тональности")
    parser.add_argument("--model-path", type=str, required=False, help="Путь к сохранённой модели (.joblib)")
    parser.add_argument("--input-path", type=str, required=False, help="Путь к CSV/XLSX с колонкой с текстом")
    parser.add_argument("--text-column", type=str, required=False, help="Имя колонки с текстом")
    parser.add_argument("--input-data-sep", type=str, required=False, help="разделитель для CSV файла")
    parser.add_argument("--output-path", type=str, required=False, help="Путь для сохранения предсказаний")
    parser.add_argument("--label-mapping", type=parse_label_mapping, required=False, help="Маппинг меток (JSON/Python literal)")
    parser.add_argument("--label-text-mapping", type=parse_label_mapping, required=False, help="Текстовая расшифровка меток")
    args = parser.parse_args()

    cfg = load_config() or {}
    paths_cfg = cfg.get("paths", {}) or {}
    columns_cfg = cfg.get("columns", {}) or {}

    default_model = paths_cfg.get("final_model_path", "./models/weights/sentiment_model.joblib")
    default_input = paths_cfg.get("test_data_path", str(Path("data") / "test.csv"))
    default_output = paths_cfg.get("predictions_path", str(Path("data") / "predictions.csv"))
    default_text_col = columns_cfg.get("text", "text")
    default_sep = ","

    model_path = args.model_path or default_model
    input_path = args.input_path or default_input
    output_path = args.output_path or default_output
    text_col = args.text_column or default_text_col
    sep = args.input_data_sep or default_sep

    label_map = args.label_mapping if args.label_mapping is not None else cfg.get("label_mapping", None)
    label_text_map = args.label_text_mapping if args.label_text_mapping is not None else cfg.get("label_text_mapping", {0: "negative", 1: "positive"})

    inp = Path(input_path)
    if not inp.exists():
        raise SystemExit(f"Input path не найден: {inp}")

    df = load_data(str(inp), return_df=True, sep=sep, check_text_column=False)

    if text_col not in df.columns:
        raise SystemExit(f"Колонка с текстом '{text_col}' не найдена. Доступные колонки: {list(df.columns)}")

    texts = df[text_col].astype(str).tolist()

    clf = SentimentClassifier(model_path=model_path)

    # Вызов predict с fallback'ами (разные сигнатуры)
    try:
        preds = clf.predict(texts, label_mapping=label_map)
    except TypeError:
        try:
            preds = clf.predict(texts, label_map)
        except TypeError:
            preds = clf.predict(texts)

    # predict_proba может отсутствовать
    try:
        probas = clf.predict_proba(texts)
    except Exception:
        probas = None

    # Приводим прогнозы/вероятности к питоновским типам и спискам длины df
    preds_py_raw = to_python_ints(preds)
    preds_list = _ensure_list_for_df(preds_py_raw, len(df))

    if probas is not None:
        probas_arr = np.asarray(probas)
        try:
            probas_py = probas_arr.tolist()
        except Exception:
            probas_py = list(probas_arr)
        probas_list = _ensure_list_for_df(probas_py, len(df))
    else:
        probas_list = [None] * len(df)

    df["prediction"] = preds_list
    # Сериализуем predict_proba в JSON-строку для CSV удобства
    df["predict_proba"] = [json.dumps(row) if row is not None else None for row in probas_list]

    # Текстовая расшифровка метки (если задана)
    if label_text_map:
        df["prediction_str"] = [_map_pred_to_text(p, label_text_map) for p in preds_list]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".xlsx":
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False, encoding='utf-8', sep=sep)

    print(f"Предсказания сохранены в {out_path}")

if __name__ == "__main__":
    main()