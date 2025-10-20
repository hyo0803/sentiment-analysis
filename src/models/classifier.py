# src/models/classifier.py
import hashlib
import numpy as np 
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from src.data.utils import preprocess_text, to_python_ints
from src.config.utils import load_config
from contextlib import contextmanager
import warnings


@contextmanager
def numpy_errstate(**kwargs):
    """
    Контекстный менеджер для временной установки numpy error handling.
    Пример: with numpy_errstate(divide='raise'): ...
    """
    old = np.geterr()
    np.seterr(**kwargs)
    try:
        yield
    finally:
        np.seterr(**old)

def safe_matmul(a, b, *, replace_zeros=False, eps=1e-12, on_error='suppress'):
    """
    Безопасное умножение a @ b с обработкой возможного divide-by-zero warning.

    Аргументы:
      a, b: numpy arrays (или объекты, поддерживающие @)
      replace_zeros (bool): если True — заменяет нулевые элементы в 'b' на eps перед умножением
      eps (float): значение для замены нулей (только при replace_zeros=True)
      on_error (str): поведение при возникновении ошибки/варнинга
          - 'suppress'  : выполнить операцию с np.seterr(divide='ignore') и вернуть результат
          - 'replace'   : попробовать заменить нули на eps и выполнить умножение
          - 'raise'     : повторно поднять ошибку FloatingPointError для дебага/логирования

    Возвращает: результат a @ b (numpy array)
    """
    try:
        return a @ b
    except FloatingPointError:
        if on_error == 'raise':
            raise
        elif on_error == 'replace' or replace_zeros:
            try:
                b_arr = np.array(b, copy=True)
                zero_mask = (b_arr == 0)
                if np.any(zero_mask):
                    b_arr[zero_mask] = eps
                return a @ b_arr
            except Exception:
                if on_error == 'raise':
                    raise
                with numpy_errstate(divide='ignore'):
                    return a @ b
        else:
            with numpy_errstate(divide='ignore'):
                return a @ b

def matmul_with_check(a, b, check_nonzero_axes=None):
    """
    Вспомогательная функция: перед matmul проверяет наличие нулей по указанным осям
    и возвращает явное предупреждение/ошибку.

    check_nonzero_axes:
      None        — не делать дополнительных проверок
      'any'       — если в b есть хоть один ноль — вернуть False и задать warning
      callable    — функция(mask) -> bool для кастомной валидации
    """
    if check_nonzero_axes is None:
        return True
    b_arr = np.array(b)
    zero_exists = np.any(b_arr == 0)
    if check_nonzero_axes == 'any':
        if zero_exists:
            warnings.warn("Found zeros in matrix 'b' before matmul — may cause divide-by-zero downstream.", RuntimeWarning)
            return False
        return True
    if callable(check_nonzero_axes):
        try:
            return bool(check_nonzero_axes(b_arr))
        except Exception:
            warnings.warn("check_nonzero_axes callable raised an exception; skipping check.", RuntimeWarning)
            return True
    return True


class SentimentClassifier:
    def __init__(self, embedder: str = None, model_path: str = None):
        """
        Инициализация: embedder и параметры берутся из config/params.yaml, если не переданы.
        """
        cfg = load_config() or {}
        training_cfg = cfg.get("training", {}) or {}
        paths_cfg = cfg.get("paths", {}) or {}

        # defaults from config
        default_embedder = training_cfg.get("embedder", "paraphrase-multilingual-MiniLM-L12-v2")
        self._default_cache = paths_cfg.get("embeddings_dir", "data/embeddings")
        # classifier default params (can be overridden in fit via classifier_kwargs)
        self._default_clf_params = training_cfg.get("classifier_params", {
            "C": 0.1,
            "penalty": "l2",
            "loss": "squared_hinge",
            "max_iter": 2000,
            "random_state": 42,
            "class_weight": None
        })

        emb = embedder or default_embedder
        # Загружаем эмбеддинг-модель один раз
        self.embedding_model = SentenceTransformer(emb)
        self.classifier = None
        self._embedding_cache = {}  # in-memory кэш: хэш → эмбеддинги
        if model_path:
            self.load_model(model_path)

    def _preprocess(self, texts):
        if isinstance(texts, str):
            return preprocess_text(texts)
        return [preprocess_text(t) for t in texts]

    def _hash_texts(self, texts):
        """Создаёт уникальный хэш для списка текстов."""
        if isinstance(texts, str):
            texts = [texts]
        combined = "|".join(sorted(texts))  # стабильный порядок
        return hashlib.md5(combined.encode("utf-8")).hexdigest()

    def _encode_with_cache(self, texts, cache_dir: str = None):
        """
        Возвращает эмбеддинги, используя:
        - in-memory кэш (быстро при повторном вызове в одном сеансе)
        - disk cache (быстро при повторном запуске скрипта)
        """
        texts_clean = self._preprocess(texts)
        cache_key = self._hash_texts(texts_clean)

        # 1. Проверяем in-memory кэш
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # 2. Проверяем disk cache
        # default cache_dir from config if not provided
        effective_cache = cache_dir if cache_dir is not None else getattr(self, "_default_cache", None)
        if effective_cache:
            cache_path = Path(effective_cache) / f"{cache_key}.npy"
            if cache_path.exists():
                embeddings = np.load(cache_path)
                self._embedding_cache[cache_key] = embeddings
                return embeddings

        # 3. Считаем эмбеддинги
        print("Векторизация текстов...")
        embeddings = self.embedding_model.encode(texts_clean, show_progress_bar=True)

        # 4. Сохраняем в оба кэша
        self._embedding_cache[cache_key] = embeddings
        if effective_cache:
            Path(effective_cache).mkdir(parents=True, exist_ok=True)
            np.save(Path(effective_cache) / f"{cache_key}.npy", embeddings)

        return embeddings

    def fit(self, texts, labels, cache_dir: str = None, classifier_kwargs: dict = None):
        """
        Обучает модель. Использует кэш эмбеддингов, чтобы избежать повторного encode.
        
        Args:
            texts: список строк или одна строка
            labels: список меток (0/1)
            cache_dir: папка для сохранения эмбеддингов на диск (None — берётся из config)
            classifier_kwargs: словарь параметров для LinearSVC (переопределяет config/params.yaml)
        """
        print("Предобработка текстов...")
        embeddings = self._encode_with_cache(texts, cache_dir=cache_dir)

        # prepare classifier params merging config defaults and provided kwargs
        clf_params = dict(self._default_clf_params or {})
        if classifier_kwargs:
            clf_params.update(classifier_kwargs)

        print(f"Обучение LinearSVC с параметрами: {clf_params}...")
        # ensure expected keys exist and pass only supported args to LinearSVC
        self.classifier = LinearSVC(**clf_params)
        self.classifier.fit(embeddings, labels)
        print("Обучение завершено.")

    def predict(self, texts, label_mapping=None):
        """
        Прогнозы меток.
        Если label_mapping задан, и представляет собой mapping исход_label->numeric (например {-1:0,1:1}),
        то функция инвертирует его и возвращает предсказания в исходном формате (числа/строки).
        Возвращает скаляр (для одиночного текста) или список предсказаний.
        """

        if self.classifier is None:
            raise ValueError("Модель не обучена и не загружена!")

        single_input = isinstance(texts, str)
        texts_clean = self._preprocess(texts)
        if single_input:
            embeddings = self.embedding_model.encode([texts_clean], show_progress_bar=False)
        else:
            embeddings = self.embedding_model.encode(texts_clean, show_progress_bar=False)

        preds = self.classifier.predict(embeddings)

        # Если задан label_mapping (оригинал->num), инвертируем её для получения num->orig
        if label_mapping:
            try:
                # нормализуем mapping: если ключи строковые цифры — приведём к int
                norm_map = {}
                for k, v in label_mapping.items():
                    nk = int(k) if isinstance(k, (str, bytes)) and str(k).lstrip('-').isdigit() else k
                    nv = int(v) if isinstance(v, (str, bytes)) and str(v).lstrip('-').isdigit() else v
                    norm_map[nk] = nv
                # invert if mapping is orig->num (values are numeric)
                vals = list(norm_map.values())
                if all(isinstance(x, (int, np.integer)) for x in vals):
                    inv = {v: k for k, v in norm_map.items()}
                    preds_mapped = [inv.get(int(p), p) for p in preds]
                else:
                    # mapping already numeric->orig or something else — try direct mapping
                    preds_mapped = [norm_map.get(p, p) for p in preds]
            except Exception:
                preds_mapped = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        else:
            # приводим к питоновским типам
            preds_mapped = preds.tolist() if hasattr(preds, "tolist") else list(preds)

        # приводим к питоновским типам перед возвратом
        out = to_python_ints(preds_mapped)
        return out[0] if single_input and isinstance(out, list) else out

    def predict_proba(self, texts):
        """
        Возвращает вероятности для обоих классов в виде списка списков [[p0,p1], ...]
        Если передан одиночный текст — возвращает список [p0,p1] (not scalar).
        """
        if self.classifier is None:
            raise ValueError("Модель не обучена и не загружена!")

        single_input = isinstance(texts, str)
        texts_clean = self._preprocess(texts)
        if single_input:
            embeddings = self.embedding_model.encode([texts_clean], show_progress_bar=False)
        else:
            embeddings = self.embedding_model.encode(texts_clean, show_progress_bar=False)

        # Получаем скор (decision_function) — LinearSVC не даёт predict_proba
        scores = self.classifier.decision_function(embeddings)
        # Преобразуем score -> prob via sigmoid; для двоичного случая формируем [1-p, p]
        sig = 1 / (1 + np.exp(-np.asarray(scores)))
        # если scores одномерный (binary)
        if sig.ndim == 1:
            probas = np.vstack([1 - sig, sig]).T
        else:
            # многоклассовый случай (unlikely for this pipeline) — нормализуем по строкам
            probas = np.array(sig)
            row_sums = probas.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            probas = probas / row_sums

        # нормализация к питоновским типам
        out = to_python_ints(probas.tolist() if hasattr(probas, "tolist") else probas)
        return out if not single_input else out[0]

    def save_model(self, path):
        """
        Сохраняет весь объект SentimentClassifier (если возможно). Для обратной совместимости
        поддерживается сохранение только sklearn-классификатора.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            # пытаемся сохранить весь объект
            joblib.dump(self, str(p))
            print(f"SentimentClassifier instance saved to {p}")
        except Exception:
            # fallback: сохраняем только sklearn-классификатор
            if self.classifier is None:
                raise ValueError("Нечего сохранять!")
            joblib.dump(self.classifier, str(p))
            print(f"Sklearn classifier saved to {p} (instance save failed)")

    def load_model(self, path):
        """
        Загружает модель. Поддерживает файлы, где сохранён весь объект SentimentClassifier,
        а также старый формат — только sklearn-классификатор.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        obj = joblib.load(str(p))
        # Если загруженный объект — экземпляр SentimentClassifier
        if hasattr(obj, "classifier") and hasattr(obj, "embedding_model"):
            # копируем состояние
            try:
                self.__dict__.update(obj.__dict__)
                print(f"SentimentClassifier instance loaded from {p}")
                return
            except Exception:
                # fallback below
                pass
        # Если загружен sklearn-классификатор (старый формат)
        self.classifier = obj
        print(f"Sklearn classifier loaded from {p} into SentimentClassifier.classifier")