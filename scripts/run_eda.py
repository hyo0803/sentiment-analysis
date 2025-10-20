#!/usr/bin/env python3
# scripts/run_eda.py
from pathlib import Path
from src.data.eda import generate_eda_report
from src.data.utils import load_data
from src.config.utils import load_config
import argparse

def main():
    parser = argparse.ArgumentParser(description="Генерация EDA-отчёта")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Путь к CSV/XLSX с данными"
    )
    parser.add_argument(
        "--input-data-sep",
        type=str,
        required=False,
        default=',',
        help="разделитель для CSV файла (например, ',' или ';')"
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Папка для отчётов"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=None,
        help="Имя колонки с текстом (по умолчанию 'text')"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Имя колонки с меткой (по умолчанию 'label')"
    )
    args = parser.parse_args()
    
    inp = Path(args.input_path)
    if not inp.exists():
        raise SystemExit(f"Input path не найден: {inp}")

    # load shared config and resolve defaults (CLI переопределяет конфиг)
    cfg = load_config() or {}
    text_col = args.text_column or cfg.get("columns", {}).get("text", "text")
    label_col = args.label_column or cfg.get("columns", {}).get("label", "label")
    report_dir = Path(args.report_dir or cfg.get("paths", {}).get("reports_dir", "data/reports"))
    report_dir.mkdir(parents=True, exist_ok=True)

    # Загрузка данных
    df = load_data(str(inp), return_df=True, sep=args.input_data_sep, check_text_column=False)

    # Проверка колонок
    if text_col not in df.columns:
        raise SystemExit(f"Колонка с текстом '{text_col}' не найдена. Доступные: {list(df.columns)}")
    if label_col not in df.columns:
        raise SystemExit(f"Колонка с меткой '{label_col}' не найдена. Доступные: {list(df.columns)}")

    # Генерация отчёта
    generate_eda_report(
        df,
        report_dir=report_dir,
        text_col=text_col,
        label_col=label_col,
        sep=args.input_data_sep
    )

    print(f"EDA отчёт сгенерирован в: {report_dir}")

if __name__ == "__main__":
    main()