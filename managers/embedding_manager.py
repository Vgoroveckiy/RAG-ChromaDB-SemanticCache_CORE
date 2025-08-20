import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import Config, config
from services.document_type_detector import DocumentType, DocumentTypeDetector
from services.document_parser import DocumentParser
from services.embedding_service import EmbeddingService


class EmbeddingManager:
    def __init__(self, config: Config):
        self.config = config
        self.type_detector = DocumentTypeDetector(config)
        self.document_parser = DocumentParser(config)
        self.embedding_service = EmbeddingService(config)
        self.splitters = self._init_splitters()

    def get_embeddings(self, doc_type: DocumentType = "default"):
        """Возвращает модель эмбеддингов для указанного типа документа"""
        return self.embedding_service.get_embeddings(doc_type)

    @property
    def embeddings(self):
        """Свойство для совместимости (возвращает модель по умолчанию)"""
        return self.embedding_service.get_embeddings("default")

    def _init_splitters(self) -> Dict[DocumentType, RecursiveCharacterTextSplitter]:
        splitters = {
            "default": RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", " "],
                length_function=len,
                add_start_index=True,
            )
        }
        for doc_type in ["legal", "qa"]:
            if doc_type in self.config.DOCUMENT_TYPE_CONFIG:
                params = self.config.DOCUMENT_TYPE_CONFIG[doc_type]
                splitters[doc_type] = RecursiveCharacterTextSplitter(
                    chunk_size=params["chunk_size"],
                    chunk_overlap=params["chunk_overlap"],
                    separators=params["separators"],
                    length_function=len,
                    add_start_index=True,
                )
        return splitters

    def get_file_hash(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        if file_path.lower().endswith(".json"):
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
                hasher.update(
                    json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
                )
        else:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        return hasher.hexdigest()

    def process_document(self, file_path: str) -> List[Document]:
        """Full document processing pipeline."""
        try:
            content = self.document_parser.parse_document(file_path)
            doc_type, reason = self.type_detector.detect(file_path, content)

            metadata = {
                "source": os.path.basename(file_path),
                "file_path": file_path,
                "file_hash": self.get_file_hash(file_path),
                "last_modified": os.path.getmtime(file_path),
                "document_type": doc_type,
                "detection_reason": reason,
            }

            return self.splitters[doc_type].create_documents([content], [metadata])
        except Exception as e:
            print(f"Error processing document {file_path}: {e}")
            return []

    def create_document_chunks(self, content: str, metadata: dict) -> List[Document]:
        """Create document chunks based on document type."""
        doc_type = metadata.get("document_type", "default")
        return self.splitters[doc_type].create_documents([content], [metadata])

    def parse_document(self, file_path: str) -> str:
        """Proxy method to document parser"""
        return self.document_parser.parse_document(file_path)

    def get_current_device(self) -> str:
        """Возвращает текущее устройство (CPU/GPU) для моделей эмбеддингов."""
        return self.embedding_service.get_current_device()
