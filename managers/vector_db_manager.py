import os
import shutil
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
import gc
import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from managers.embedding_manager import EmbeddingManager
from config import Config


class VectorDatabase:
    """Class for managing ChromaDB vector databases with enhanced error handling"""

    def __init__(
        self, db_path: str, cache_path: str, embedding_manager: EmbeddingManager
    ) -> None:
        self.db_path = db_path
        self.cache_path = cache_path
        self.embedding_manager = embedding_manager
        self.db: Optional[Chroma] = None
        self.cache_db: Optional[Chroma] = None

    def load_or_create(self, force_recreate: bool = False) -> None:
        """Load or create main ChromaDB collection"""
        try:
            if force_recreate and os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
                print(f"Удалена существующая база: {self.db_path}")

            device = self.embedding_manager.get_current_device()
            self.db = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_manager.embeddings,
                collection_name="documents_collection",
            )
            print(f"✅ База данных инициализирована. Устройство эмбеддингов: {device}")

            # Test functionality
            test_id = "test_" + str(uuid.uuid4())
            self.db.add_texts(
                texts=["test document"],
                metadatas=[{"source": "system", "test": True}],
                ids=[test_id],
            )
            self.db.delete(ids=[test_id])

        except Exception as e:
            print(f"❌ Ошибка инициализации базы данных: {e}")
            try:
                self.db = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embedding_manager.embeddings,
                    collection_name="documents_collection",
                )
                print("✅ База данных инициализирована (fallback)")
            except Exception as e:
                raise RuntimeError(f"Не удалось инициализировать базу данных: {e}")

    def load_or_create_cache(self, force_recreate: bool = False) -> None:
        """Load or create cache ChromaDB collection"""
        if force_recreate:
            if os.path.exists(self.cache_path):
                shutil.rmtree(self.cache_path)
            else:
                print(f"Cache folder not found ({self.cache_path}). Creating new.")

        self.cache_db = Chroma(
            persist_directory=self.cache_path,
            embedding_function=self.embedding_manager.embeddings,
            collection_name="current",
        )

        if self.cache_db and hasattr(self.cache_db, "_collection"):
            count = self.cache_db._collection.count()
            if count > 0:
                print(f"Cache collection loaded ({count} items)")
            else:
                print("New cache collection created")

    def add_to_cache(
        self, question: str, answer: str, sources: Optional[List[str]] = None
    ) -> bool:
        """Add question-answer pair to semantic cache"""
        try:
            if not self.cache_db:
                self.load_or_create_cache()

            self.cleanup_expired_cache_entries(
                self.embedding_manager.config.CACHE_TTL_DAYS
            )

            doc_id = str(uuid.uuid4())
            metadata = {
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
                "sources": json.dumps(sources) if sources else "[]",
                "doc_id": doc_id,
            }

            self.cache_db.add_texts(
                texts=[question], metadatas=[metadata], ids=[doc_id]
            )
            return True
        except Exception as e:
            print(f"Error adding to cache: {e}")
            return False

    def get_cached_answer(
        self,
        question: str,
        similarity_threshold: float = 0.85,
    ) -> Optional[str]:
        """Search for cached answer to similar question"""
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")

        if not self.cache_db:
            return None

        try:
            results = self.cache_db.similarity_search_with_score(question, k=1)
            if results:
                doc, score = results[0]
                if score <= similarity_threshold:
                    return str(doc.metadata.get("answer"))
        except Exception as e:
            print(f"Cache search error: {e}")
        return None

    def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents from ChromaDB by their IDs"""
        if not isinstance(doc_ids, list):
            raise TypeError("doc_ids must be a list")
        if not doc_ids:
            return

        if not self.db:
            print("ChromaDB not initialized")
            return

        try:
            self.db.delete(ids=doc_ids)
        except Exception as e:
            raise RuntimeError(f"Error deleting documents: {e}")

    def delete_cached_entries_by_source(self, source_file_name: str) -> None:
        """Delete cache entries associated with specified source file"""
        if not isinstance(source_file_name, str):
            raise TypeError("source_file_name must be a string")

        if not self.cache_db:
            return

        try:
            collection = self.cache_db._collection
            all_entries = collection.get(include=["metadatas"])

            if all_entries and "ids" in all_entries and "metadatas" in all_entries:
                ids_to_delete = []
                for i, doc_id in enumerate(all_entries["ids"]):
                    if i < len(all_entries["metadatas"]):
                        metadata = all_entries["metadatas"][i]
                        if isinstance(metadata, dict):
                            sources = json.loads(metadata.get("sources", "[]") or "[]")
                            if (
                                isinstance(sources, list)
                                and source_file_name in sources
                            ):
                                ids_to_delete.append(doc_id)

                if ids_to_delete:
                    self.cache_db.delete(ids=ids_to_delete)
        except Exception as e:
            raise RuntimeError(f"Error processing cache: {e}")

    def cleanup_expired_cache_entries(self, ttl_days: float) -> None:
        """Clean up expired entries from semantic cache"""
        if not self.cache_db:
            return

        try:
            collection = self.cache_db._collection
            current_time = datetime.now()
            expiration_threshold = current_time - timedelta(days=ttl_days)

            if collection:
                all_entries = collection.get(include=["metadatas"])
                if all_entries and "ids" in all_entries and "metadatas" in all_entries:
                    ids_to_delete = []
                    for i, doc_id in enumerate(all_entries["ids"]):
                        if i < len(all_entries["metadatas"]):
                            metadata = all_entries["metadatas"][i]
                            if isinstance(metadata, dict):
                                timestamp_str = metadata.get("timestamp", "")
                                if isinstance(timestamp_str, str):
                                    try:
                                        cache_datetime = datetime.fromisoformat(
                                            timestamp_str
                                        )
                                        if cache_datetime < expiration_threshold:
                                            ids_to_delete.append(doc_id)
                                    except ValueError:
                                        continue

                    if ids_to_delete:
                        self.cache_db.delete(ids=ids_to_delete)
        except Exception as e:
            raise RuntimeError(f"Error cleaning cache: {e}")

    def get_documents_by_hash(self, file_hash: str) -> dict:
        """Retrieve documents by file hash from ChromaDB"""
        if not self.db:
            return {}

        try:
            return self.db.get(
                where={"file_hash_full": file_hash}, include=["metadatas", "documents"]
            )
        except Exception as e:
            print(f"Error getting documents by hash: {e}")
            return {}

    def get_all_metadata(self, batch_size: int = 1000) -> List[Dict]:
        """Retrieve all metadata from ChromaDB with pagination."""
        if not self.db:
            return []

        metadatas = []
        offset = 0
        while True:
            items = self.db._collection.get(
                include=["metadatas"], limit=batch_size, offset=offset
            )
            if not items["metadatas"]:
                break
            metadatas.extend(items["metadatas"])
            offset += batch_size
        return metadatas

    def close(self):
        """Гарантированное освобождение ресурсов"""
        try:
            if self.db:
                self.db = None
            if self.cache_db:
                self.cache_db = None
        except Exception as e:
            print(f"Ошибка при закрытии VectorDatabase: {e}")
        finally:
            gc.collect()
