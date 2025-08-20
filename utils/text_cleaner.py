import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN


class LegalTextCleaner:
    def __init__(self):
        # Загрузка модели для русских юридических текстов на CPU
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            device="cpu",  # Принудительное использование CPU
        )

    def clean(self, text):
        """Основной метод очистки юридических текстов"""
        # Предварительная обработка: удаление конкретных артефактов
        preprocessed = self.preprocess(text)

        # Разбивка на предложения
        sentences = self.split_sentences(preprocessed)

        # Кластеризация для выделения основного контента
        return self.cluster_sentences(sentences)

    def preprocess(self, text):
        """Удаление только конкретных артефактов"""
        patterns = [
            r"Страница \d+ из \d+",  # номера страниц
            r"Документ предоставлен КонсультантПлюс",
            r"Дата сохранения: \d{2}\.\d{2}\.\d{4}",
            r"www\.consultant\.ru",
            r"КонсультантПлюс.*?поддержка",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        return text

    def split_sentences(self, text):
        """Разбивка текста на предложения"""
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def cluster_sentences(self, sentences):
        """Кластеризация предложений для выявления основного контента"""
        if len(sentences) < 3:
            return " ".join(sentences)

        embeddings = self.model.encode(sentences)
        clustering = DBSCAN(eps=0.7, min_samples=2).fit(embeddings)

        # Выбор основного кластера
        labels = clustering.labels_
        if len(set(labels)) > 1:  # Если есть несколько кластеров
            main_cluster = max(set(labels), key=list(labels).count)
            return " ".join(
                sent for sent, label in zip(sentences, labels) if label == main_cluster
            )
        return " ".join(sentences)
