# scripts/normalize_data_struct.py
import argparse
import json
from src.data.utils import normalize_dataset

def parse_label_mapping(mapping_str):
    """Парсит строку вида '{"negative":0, "positive":1}' в dict."""
    if mapping_str is None:
        return None
    try:
        return json.loads(mapping_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Неверный формат label_mapping: {e}")

def main():
    parser = argparse.ArgumentParser(description="Нормализация датасета под стандартный формат")
    parser.add_argument("--input-path", required=True, help="Путь к исходному CSV/XLSX")
    parser.add_argument("--output-path", required=True, help="Путь для нормализованного CSV/XLSX")
    parser.add_argument("--text-column", required=True, help="Имя колонки с текстом в исходнике")
    parser.add_argument("--label-column", required=True, help="Имя колонки с меткой в исходнике")
    parser.add_argument(
        "--label-mapping",
        type=parse_label_mapping,
        help='JSON-словарь для маппинга меток, например: \'{"positive":1,"negative":0}\''
    )
    parser.add_argument( "--input-data-sep",
        type=str,
        required=False,
        help="разделитель для CSV файла (например, ',' или ';')")
    parser.add_argument( "--output-data-sep",
        type=str,
        required=False,
        help="разделитель для выходного CSV файла (например, ',' или ';')")
    
    args = parser.parse_args()
    
    normalize_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        text_column=args.text_column,
        label_column=args.label_column,
        label_mapping=args.label_mapping,
        input_sep=args.input_data_sep if args.input_data_sep else ',',
        output_sep=args.output_data_sep if args.output_data_sep else ','
    )

if __name__ == "__main__":
    main()