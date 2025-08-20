from typing import Dict, Literal
from config import Config
from langchain_huggingface import HuggingFaceEmbeddings
from utils import gpu_utils

DocumentType = Literal["legal", "qa", "default"]


class EmbeddingService:
    def __init__(self, config: Config):
        self.config = config
        self._models = {}  # Для ленивой загрузки моделей

    def get_embeddings(
        self, doc_type: DocumentType = "default"
    ) -> HuggingFaceEmbeddings:
        """Возвращает модель эмбеддингов для указанного типа документа"""
        if doc_type not in self._models:
            self._models[doc_type] = self._load_model(doc_type)
        return self._models[doc_type]

    def _load_model(self, doc_type: DocumentType = "default") -> HuggingFaceEmbeddings:
        """Загружает модель эмбеддингов с проверкой доступности GPU"""
        # Определяем имя модели для типа документа
        model_name = self.config.EMBEDDING_MODEL
        if doc_type in self.config.DOCUMENT_TYPE_CONFIG:
            model_name = self.config.DOCUMENT_TYPE_CONFIG[doc_type].get(
                "model", model_name
            )

        # Проверяем доступность GPU
        use_gpu = gpu_utils.gpu_available(500)  # 500MB минимальный запас
        device = "cuda" if use_gpu else "cpu"
        print(
            f"Загрузка модели эмбеддингов '{model_name}' для типа '{doc_type}' на устройство: {device}"
        )

        return HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs={"device": device}
        )

    def get_current_device(self) -> str:
        """Возвращает текущее устройство (CPU/GPU) для моделей эмбеддингов."""
        # Проверяем, есть ли хотя бы одна модель, загруженная на GPU
        for model in self._models.values():
            if model.model_kwargs.get("device", "cpu") == "cuda":
                return "cuda"
        return "cpu"
