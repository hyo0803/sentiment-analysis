# scripts/run_eda.py
import argparse
from src.data.eda import generate_eda_report

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
        help="разделитель для CSV файла (например, ',' или ';')"
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="data/reports",
        help="Папка для отчётов"
    )
    args = parser.parse_args()
    
    generate_eda_report(args.input_path, 
                        report_dir=args.report_dir, 
                        sep=args.input_data_sep if args.input_data_sep else ',')

if __name__ == "__main__":
    main()