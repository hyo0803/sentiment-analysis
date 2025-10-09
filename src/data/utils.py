import pandas as pd # type: ignore
from pathlib import Path
import re

def load_data(
    data_path: str,
    return_df: bool = False, 
    sep=',', 
    check_text_column: bool = True
    ):
    """Загружает данные из CSV/XLSX.
    
    Args:
        data_path: путь к файлу для считывания (CSV или XLSX).
        return_df: если True — возвращает DataFrame, иначе кортеж списков (texts, labels).
        sep: разделитель для CSV файла (например, ',' или ';').
        check_text_column: если True — проверяет наличие колонок 'text' и 'label'.
        
    Returns:
        DataFrame или кортеж списков (texts, labels).
    """
    if data_path.lower().endswith('.csv'):
        try:
            df = pd.read_csv(data_path, sep=sep)
        except:
            delims = [',', ';', '\t', '|']
            delims.remove(sep)
            for delim in delims:
                try:
                    df = pd.read_csv(data_path, sep=delim)
                    print(f"Warning: Использован альтернативный разделитель '{delim}' для {data_path}")
                    break
                except:
                    continue
            else:
                raise ValueError(f"Не удалось прочитать {data_path} с известными разделителями {delims + [sep]}")
    
    elif data_path.lower().endswith('.xlsx'):
        df = pd.read_excel(data_path)
        
    else:
        raise ValueError("Поддерживаемые форматы файлов: .csv, .xlsx")
    
    if check_text_column:
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Файл должен содержать колонки 'text' и 'label'")
    
    if return_df:
        return df
    return df["text"].tolist(), df["label"].tolist()

def preprocess_text(text: str) -> str:
    """
    Очищает текст от управляющих символов и лишних пробелов (минимальная очистка).
    - Удаляет переносы строк, табуляции, начальные спецсимволы
    - Нормализует пробелы
    - Сохраняет всё важное: пунктуацию, эмодзи, хештеги, регистр
    
    Args:
        text (str): Входной текст.
        
    Returns:
        str: Очищенный текст.
    """
    if not isinstance(text, str):
        text = str(text)
    # Удаляем \n, \t и другие control-символы (кроме пробела)
    text = re.sub(r"[\n\t\r]+", " ", text)
    # Убираем множественные пробелы
    text = re.sub(r"\s+", " ", text)
    # Убираем пробелы по краям
    return text.strip()

def normalize_dataset(
    input_path: str,
    output_path: str,
    text_column: str,
    label_column: str,
    label_mapping: dict = {"negative":0, "positive":1},
    input_sep: str = ',',
    output_sep: str = ','
) -> pd.DataFrame:
    """
    Нормализует датасет: переименовывает колонки и (опционально) маппит метки в 0/1.
    
    Args:
        input_path (str): Путь к исходному CSV.
        output_path (str): Путь для сохранения нормализованного CSV.
        text_column (str): Имя колонки с текстом в исходном датасете.
        label_column (str): Имя колонки с меткой в исходном датасете.
        label_mapping (dict, optional): Словарь для преобразования меток.
            Пример: {"negative": 0, "positive": 1}
            Если None — оставляет метки как есть (но проверяет, что они 0/1 или уже числа).
        input_sep (str): Разделитель для чтения CSV файла (например, ',' или ';').
        output_sep (str): Разделитель для записи данных в CSV файл (например, ',' или ';').
    
    Returns:
        None. Сохраняет результат в output_path.
    """
    # Загрузка
    df = load_data(input_path, return_df=True, sep=input_sep, check_text_column=False)
    
    # Проверка наличия колонок + Выбираем нужные колонки и создаём стандартные имена
    if text_column not in df.columns:
        raise ValueError(f"Колонка '{text_column}' не найдена в {input_path}")
    else:
        df["text"] = df[text_column]
        
    if label_column not in df.columns:
        raise ValueError(f"Колонка '{label_column}' не найдена в {input_path}")
    else:
        df["label"] = df[label_column]

    df = df[df['label'].isin([label_mapping['negative'], label_mapping['positive']])]
    
    # Сохранение
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, sep=output_sep, encoding='utf-8')
    print(f"Нормализованный датасет сохранён: {output_path}")
    print(f"Примеры:\n{df.head()}")
    
    return df