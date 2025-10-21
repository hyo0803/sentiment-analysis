import pandas as pd # type: ignore
from pathlib import Path
import re
import numpy as np # type: ignore
from contextlib import contextmanager
import warnings
import json
import ast
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

# NOTE:
# Utility functions for safe matmul / numpy errstate were moved to
# src/models/classifier.py (numpy_errstate, safe_matmul, matmul_with_check).
# Removed forwarding wrappers from utils to avoid duplication and unused exports.

def load_data(
    data_path: Union[str, Path],
    return_df: bool = False,
    sep: str = ',',
    check_text_column: bool = True
) -> Union[pd.DataFrame, Tuple[List[str], List[Any]]]:
    """
    Загружает данные из CSV/XLSX или директории (в этом случае выбирает первый найденный CSV/XLSX).

    Args:
        data_path: путь к файлу или директории.
        return_df: если True — возвращает DataFrame, иначе кортеж (texts, labels).
        sep: разделитель для CSV файла (например, ',' или ';').
        check_text_column: если True — проверяет наличие колонок 'text' и 'label'.

    Returns:
        DataFrame или кортеж (texts, labels).
    """
    p = Path(data_path)
    if p.is_dir():
        # Ищем первые подходящие файлы в папке
        cand = next(p.glob("*.csv"), None) or next(p.glob("*.xlsx"), None)
        if cand is None:
            raise FileNotFoundError(f"No CSV/XLSX files found in directory: {p}")
        p = cand

    if str(p).lower().endswith('.csv'):
        # Попытка чтения с указанным sep, затем пробуем другие разделители
        try:
            df = pd.read_csv(p, sep=sep)
        except Exception:
            tried = [sep]
            delims = [',', ';', '\t', '|']
            if sep in delims:
                delims.remove(sep)
            for delim in delims:
                try:
                    df = pd.read_csv(p, sep=delim)
                    print(f"Warning: Использован альтернативный разделитель '{delim}' для {p}")
                    tried.append(delim)
                    break
                except Exception:
                    tried.append(delim)
                    continue
            else:
                raise ValueError(f"Не удалось прочитать {p} с разделителями {tried}")
    elif str(p).lower().endswith('.xlsx') or str(p).lower().endswith('.xls'):
        df = pd.read_excel(p)
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

def encode_labels(labels, label_map=None, clear_extra_classes=False):
    """
    Кодирует бинарные метки и возвращает (encoded_labels, valid_mask).

    Если clear_extra_classes=True — encoded_labels возвращается уже отфильтрованным
    (только для записей, метки которых присутствуют в label_map), valid_mask остаётся
    маской по отношению к исходному массиву (True = оставляем).
    """
    labels_arr = np.asarray(labels)
    n = len(labels_arr)
    valid_mask = np.ones(n, dtype=bool)

    if label_map is not None:
        # Определим направление маппинга — хотим получить map_from_original->numeric
        keys = list(label_map.keys())
        vals = list(label_map.values())

        keys_are_str = all(isinstance(k, str) for k in keys)
        keys_are_num = all(isinstance(k, (int, float)) for k in keys)
        vals_are_str = all(isinstance(v, str) for v in vals)
        vals_are_num = all(isinstance(v, (int, float)) for v in vals)

        final_map = None

        if keys_are_str and vals_are_num:
            # {'neg':0, 'pos':1}
            final_map = {k: int(v) for k, v in label_map.items()}
        elif keys_are_num and vals_are_num:
            # {-1:0, 1:1}  — ключи numeric (исходные), значения numeric (целевые)
            final_map = {k: int(v) for k, v in label_map.items()}
        elif keys_are_num and vals_are_str:
            # {0:'neg',1:'pos'} — инвертируем, хотим string->numeric
            inv = {v: int(k) for k, v in label_map.items()}
            final_map = inv
        else:
            raise ValueError("Неподдерживаемый формат label_map. Ожидаются string->num, num->num или num->string (для инверсии).")

        # Определяем валидность и кодируем (np.nan для невалидных)
        encoded = np.full(n, np.nan, dtype=float)
        for i, v in enumerate(labels_arr):
            if v in final_map:
                encoded[i] = final_map[v]
            else:
                valid_mask[i] = False

        # Собираем уникальные значения только для валидных позиций
        valid_encoded = np.unique(encoded[valid_mask]) if np.any(valid_mask) else np.array([])
        
        if not np.all(valid_mask):
            if clear_extra_classes:
                # Отфильтруем теперь только валидные значения и продолжим нормализацию на них
                filtered_encoded = encoded[valid_mask]
                # Нормализуем filtered_encoded к {0,1} если требуется
                unique_filtered = np.unique(filtered_encoded)
                if unique_filtered.size == 2:
                    if set(unique_filtered) != {0, 1}:
                        remap = {unique_filtered[0]: 0, unique_filtered[1]: 1}
                        for idx in range(filtered_encoded.size):
                            filtered_encoded[idx] = remap[filtered_encoded[idx]]
                elif unique_filtered.size != 2:
                    raise RuntimeError(f"После фильтрации получено {unique_filtered.size} уникальных значений (ожидалось 2): {unique_filtered}")
                # Возвращаем отфильтрованный массив и маску (для дальнейшей фильтрации DataFrame)
                return filtered_encoded, valid_mask
            else:
                # Если не очищаем лишние классы, выбрасываем ошибку
                invalid_vals = set(labels_arr[~valid_mask])
                raise ValueError(f"Обнаружены метки, отсутствующие в label_map: {invalid_vals}. Укажите clear_extra_classes=True, чтобы удалить такие записи.")

        # Если все метки валидны — продолжаем как раньше
        if valid_encoded.size == 2:
            if set(valid_encoded) == {0, 1}:
                return encoded, valid_mask
            else:
                remap = {valid_encoded[0]: 0, valid_encoded[1]: 1}
                for i in np.where(valid_mask)[0]:
                    encoded[i] = remap[encoded[i]]
                return encoded, valid_mask
        else:
            raise RuntimeError(f"После кодирования получено {valid_encoded.size} уникальных значений (ожидалось 2): {valid_encoded}")

    else:
        # label_map == None -> пытаемся инференсить бинарную задачу
        unique_vals = np.unique(labels_arr)
        if unique_vals.size == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            encoded = np.array([mapping[x] for x in labels_arr], dtype=int).astype(float)
            valid_mask = np.ones(n, dtype=bool)
            return encoded, valid_mask
        elif unique_vals.size < 2:
            raise ValueError(f"Обнаружен {unique_vals.size} класс(ов). Требуется минимум 2 класса для бинарной задачи.")
        else:
            raise ValueError(f"Обнаружено более 2 классов: {unique_vals}. Укажите label_map или установите clear_extra_classes=True вместе с подходящим label_map.")


def normalize_dataset(
    text_column: str,
    label_column: str,
    input_path: str = None,
    output_path: str = None,
    data: pd.DataFrame = None,
    label_mapping: dict = None,
    clear_extra_classes: bool = False,
    input_sep: str = ',',
    output_sep: str = ','
) -> pd.DataFrame:
    """
    Нормализует датасет: переименовывает колонки и (опционально) маппит метки в 0/1.

    Поведение при clear_extra_classes:
      - True: удаляет строки с метками, не входящими в label_mapping (если задан),
              или оставляет только два класса если label_mapping=None (но в таком случае лучше задать label_mapping).
      - False: при обнаружении неизвестных меток — выбрасывается ошибка.
    """
    if data is None and input_path:
        # Загрузка
        df = load_data(input_path, return_df=True, sep=input_sep, check_text_column=False)
    elif isinstance(data, pd.DataFrame) and data.empty==False and not input_path:
        df = data
    else:
        raise FileNotFoundError("Данные не найдены! Проверьте корректность пути файла или корректность датафрейма")
    
    # Проверка наличия колонок + Выбираем нужные колонки и создаём стандартные имена
    if text_column not in df.columns:
        raise ValueError(f"Колонка '{text_column}' не найдена в {input_path}")
    else:
        df["text"] = df[text_column]
        
    if label_column not in df.columns:
        raise ValueError(f"Колонка '{label_column}' не найдена в {input_path}")
    else:
        original_labels = df[label_column].values

    # Кодирование меток (получаем массив и маску)
    encoded, valid_mask = encode_labels(original_labels, label_map=label_mapping, clear_extra_classes=clear_extra_classes)

    # Обработка невалидных записей
    if not valid_mask.all():
        if clear_extra_classes:
            removed = (~valid_mask).sum()
            # Фильтруем DataFrame по маске
            df = df.loc[valid_mask].reset_index(drop=True)
            # encoded может уже быть отфильтрован (encode_labels возвращал filtered) или того же размера, что и исходные метки
            if len(encoded) == len(valid_mask):
                # encoded ещё не был отфильтрован — применяем маску
                encoded = encoded[valid_mask]
            # иначе encoded уже отфильтрован — ничего не делаем
            print(f"Удалено {removed} строк с неизвестными метками.")
        else:
            invalid_vals = set(original_labels[~valid_mask])
            raise ValueError(f"Обнаружены метки, отсутствующие в label_mapping: {invalid_vals}")

    # Присваиваем финальную колонку label (уже приведена к 0/1 внутри encode_labels)
    df["label"] = encoded.astype(int)

    # Сохранение
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, sep=output_sep, encoding='utf-8')
        print(f"Нормализованный датасет сохранён: {output_path}")
    print(f"Примеры:\n{df.head()}")

    return df

def to_python_ints(x):
    """
    Преобразует numpy-скаляры/массивы и питон-скаляры в питоновские примитивы:
      - None -> None
      - numpy / python scalar -> int / float / str
      - array-like -> list[int|float|...]
    Поведение:
      - Если элемент выглядит как целое число (np.integer или int) -> int
      - Если элемент — float -> float
      - Иначе возвращается как есть (например, str)
    """
    if x is None:
        return None

    # Быстрая обработка питон-скаляров
    if isinstance(x, (int, float, str)):
        return int(x) if isinstance(x, int) else float(x) if isinstance(x, float) else x

    # Преобразуем в numpy array для унификации
    try:
        arr = np.asarray(x)
    except Exception:
        # если не удалось привести к массиву — вернём исходное значение
        return x

    # Скаляры (numpy.scalar или 0-d array)
    if arr.ndim == 0:
        val = arr.item()
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, (np.floating, float)):
            return float(val)
        return val

    # Массив/итерабель — приводим к списку и конвертируем элементы
    lst = arr.tolist()
    def _conv(v):
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return float(v)
        return v

    # Если lst сам по себе является не-итерируемым примитив (защита), вернуть преобразованный элемент
    if not isinstance(lst, list):
        return _conv(lst)

    return [_conv(el) for el in lst]

def parse_label_mapping(mapping_str: Optional[Union[str, Dict[Any, Any]]]):
    """
    Парсит строку/объект с маппингом меток.
    Поддерживает:
      - dict (возвращается как есть после нормализации ключей)
      - корректный JSON: '{"-1":0, "1":1}'
      - Python-литерал: '{-1:0, 1:1}' (через ast.literal_eval)

    Возвращает dict или None. Поднимает argparse.ArgumentTypeError при неверном формате.
    """
    if mapping_str is None:
        return None

    # Если уже dict — используем его
    if isinstance(mapping_str, dict):
        mapping = mapping_str
    else:
        # Попытка JSON (строгий формат), затем ast.literal_eval
        try:
            mapping = json.loads(mapping_str)
        except Exception:
            try:
                mapping = ast.literal_eval(mapping_str)
            except Exception as e:
                raise argparse.ArgumentTypeError(f"Неверный формат label_mapping: {e}")

    if not isinstance(mapping, dict):
        raise argparse.ArgumentTypeError("label_mapping должен быть словарём, например: '{-1:0, 1:1}' или '{\"-1\":0,\"1\":1}'")

    # Нормализация: попытаться преобразовать строковые цифровые ключи/значения в int
    norm: Dict[Any, Any] = {}
    for k, v in mapping.items():
        nk = k
        nv = v
        if isinstance(k, str):
            if k.lstrip('-').isdigit():
                try:
                    nk = int(k)
                except Exception:
                    pass
        if isinstance(v, str):
            if v.lstrip('-').isdigit():
                try:
                    nv = int(v)
                except Exception:
                    pass
        norm[nk] = nv
    return norm

def ensure_list_for_df(x, length):
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

def map_pred_to_text(pred, mapping):
    """Безопасная текстовая расшифровка метки через mapping."""
    if mapping is None:
        return str(pred)
    try:
        key = int(pred)
    except Exception:
        key = pred
    return mapping.get(key, mapping.get(str(key), str(pred)))
