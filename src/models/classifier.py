# src/models/classifier.py
import numpy as np # type: ignore
import joblib # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from sklearn.svm import LinearSVC # type: ignore

from src.data.utils import preprocess_text

class SentimentClassifier:
    def __init__(self, embedder='paraphrase-multilingual-MiniLM-L12-v2', model_path=None):
        self.embedding_model = SentenceTransformer(embedder)
        self.classifier = None
        if model_path:
            self.load_model(model_path)
    

    def _preprocess(self, texts):
        """Применяет preprocess_text к одному или списку текстов."""
        if isinstance(texts, str):
            return preprocess_text(texts)
        return [preprocess_text(t) for t in texts]
    
    
    def fit(self, texts, labels):
        print("Предобработка текстов...")
        texts_clean = self._preprocess(texts)
        print("Векторизация...")
        embeddings = self.embedding_model.encode(texts_clean, show_progress_bar=True)
        print("Обучение LinearSVC...")
        self.classifier = LinearSVC(
            C=0.1,
            penalty='l2',
            loss='squared_hinge',
            max_iter=2000,
            random_state=42,
            class_weight=None
        )
        self.classifier.fit(embeddings, labels)
        print("Обучение завершено.")
    
    
    def predict(self, texts):
        if self.classifier is None:
            raise ValueError("Модель не обучена и не загружена!")
        
        single_input = isinstance(texts, str)
        texts_clean = self._preprocess(texts)
        if single_input:
            embeddings = self.embedding_model.encode([texts_clean], show_progress_bar=False)
        else:
            embeddings = self.embedding_model.encode(texts_clean, show_progress_bar=False)
        
        predictions = self.classifier.predict(embeddings)
        
        # Преобразуем числовые метки в строки
        label_map = {0: "negative", 1: "positive"}
        if len(predictions) > 0 and isinstance(predictions[0], (int, np.integer)):
            predictions = [label_map.get(p, p) for p in predictions]
        
        return predictions[0] if single_input else predictions
    
    
    def predict_proba(self, texts):
        if self.classifier is None:
            raise ValueError("Модель не обучена и не загружена!")
        
        single_input = isinstance(texts, str)
        texts_clean = self._preprocess(texts)
        if single_input:
            embeddings = self.embedding_model.encode([texts_clean], show_progress_bar=False)
        else:
            embeddings = self.embedding_model.encode(texts_clean, show_progress_bar=False)
        
        scores = self.classifier.decision_function(embeddings)
        proba = 1 / (1 + np.exp(-scores))
        return float(proba[0]) if single_input else proba.tolist()
    
    
    def save_model(self, path):
        if self.classifier is None:
            raise ValueError("Нечего сохранять!")
        joblib.dump(self.classifier, path)
        print(f"Модель сохранена в {path}")
    
    
    def load_model(self, path):
        self.classifier = joblib.load(path)
        print(f"Модель загружена из {path}")