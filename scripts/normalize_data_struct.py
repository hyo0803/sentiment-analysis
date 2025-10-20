#!/usr/bin/env python3
# scripts/normalize_data_struct.py
import argparse
from pathlib import Path
from src.data.utils import normalize_dataset, parse_label_mapping
from src.config.utils import load_config

def main():
    # загрузка конфигурации проекта (config/params.yaml)
    cfg = load_config() or {}
    columns_cfg = cfg.get("columns", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}

    default_text_col = columns_cfg.get("text", "text")
    default_label_col = columns_cfg.get("label", "label")
    default_label_mapping = cfg.get("label_mapping", None)
    cfg_norm_path = paths_cfg.get("normalized_data_path", "")
    cfg_clear = cfg.get("clear_extra_classes", True)

    parser = argparse.ArgumentParser(description="Нормализация датасета под стандартный формат")
    parser.add_argument("--input-path", required=True, help="Путь к исходному CSV/XLSX")
    parser.add_argument("--output-path", required=False, help="Путь для нормализованного CSV/XLSX")
    parser.add_argument("--text-column", required=False, default=None, help=f"Имя колонки с текстом (по умолч. из конфига: {default_text_col})")
    parser.add_argument("--label-column", required=False, default=None, help=f"Имя колонки с меткой (по умолч. из конфига: {default_label_col})")
    parser.add_argument("--label-mapping", type=parse_label_mapping, required=False, default=None,
                        help='JSON или Python-литерал для маппинга меток, напр. \'{"positive":1,"negative":0}\' или "{-1:0,1:1}"')
    # mutually exclusive flags to explicitly set/clear behavior; default remains None if not provided
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--clear-extra-classes", dest="clear_extra_classes", action="store_true", default=None,
                       help="Удалять строки с метками, не входящими в label_mapping (переопределяет config).")
    group.add_argument("--no-clear-extra-classes", dest="clear_extra_classes", action="store_false", default=None,
                       help="Не удалять строки с неизвестными метками (переопределяет config).")
    parser.add_argument("--input-data-sep", type=str, required=False, default=',',
                        help="разделитель для входного CSV (например ',' или ';')")
    parser.add_argument("--output-data-sep", type=str, required=False, default=',',
                        help="разделитель для выходного CSV (например ',' или ';')")

    args = parser.parse_args()

    inp = Path(args.input_path)
    if not inp.exists():
        parser.error(f"Input path не найден: {inp}")

    # resolve effective parameters (CLI overrides config)
    text_col = args.text_column or default_text_col
    label_col = args.label_column or default_label_col
    label_mapping = args.label_mapping if args.label_mapping is not None else default_label_mapping
    clear_extra = args.clear_extra_classes if args.clear_extra_classes is not None else cfg_clear

    # determine output path: CLI -> config.paths.normalized_data_path -> norm_<input_name>
    if args.output_path:
        out = Path(args.output_path)
    elif cfg_norm_path:
        out = Path(cfg_norm_path)
    else:
        out = inp.parent / f"norm_{inp.name}"

    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        normalize_dataset(
            input_path=str(inp),
            output_path=str(out),
            text_column=text_col,
            label_column=label_col,
            label_mapping=label_mapping,
            clear_extra_classes=clear_extra,
            input_sep=args.input_data_sep,
            output_sep=args.output_data_sep
        )
        print(f"Нормализация успешно выполнена. Файл сохранён: {out}")
    except Exception as e:
        print(f"Ошибка при нормализации: {e}")
        raise

if __name__ == "__main__":
    main()