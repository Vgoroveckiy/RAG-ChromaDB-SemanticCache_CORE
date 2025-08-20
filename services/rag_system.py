import gc
import os
import shutil
from typing import List, Optional, Dict
import utils.gpu_utils as gpu_utils
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from core.llm_manager import create_llm_provider

from config import Config, config
from managers.embedding_manager import EmbeddingManager
from managers.vector_db_manager import VectorDatabase
from services.indexing_service import (
    parse_files,
    cleanup_deleted_files,
)


class RAGSystem:
    """Main RAG system class."""

    def __init__(self, config: Config) -> None:
        if not isinstance(config, Config):
            raise TypeError("config must be a Config instance")

        self.config = config
        self.embedding_manager = EmbeddingManager(config)
        self.vector_db = VectorDatabase(
            config.CHROMA_DB_PATH, config.CHROMA_CACHE_PATH, self.embedding_manager
        )
        self.llm_provider = None  # Инициализация перенесена в initialize()
        self.llm = None
        self.qa_chain = None

    def initialize(self):
        """Initialize RAG system components."""
        # Инициализация векторной базы и других компонентов
        self.vector_db.load_or_create()
        self.vector_db.load_or_create_cache()
        cleanup_deleted_files(self.vector_db, self.config.INPUT_DIR)
        parse_files(self.config.INPUT_DIR, self.vector_db)
        print("Инициализация завершена")

        # Инициализируем LLM провайдер
        self.llm_provider = create_llm_provider(self.config.__dict__)
        self.llm = self.llm_provider.get_llm()
        self._init_chains()

        print(f"Инициализирован провайдер LLM: {self.config.LLM_PROVIDER}")
        print(f"Используемая модель: {self.llm_provider.get_model_name()}")

    def get_available_doc_types(self) -> List[str]:
        """Return list of document types present in the database"""
        if not self.vector_db.db or not hasattr(self.vector_db.db, "_collection"):
            return []

        try:
            items = self.vector_db.db._collection.get(include=["metadatas"])
            if not items or not isinstance(items, dict):
                return []

            types = set()
            for meta in items.get("metadatas", []):
                if isinstance(meta, dict) and "document_type" in meta:
                    types.add(meta["document_type"])

            return list(types) if types else ["default"]
        except Exception as e:
            print(f"Error getting document types: {e}")
            return []

    def _init_chains(self):
        """Инициализация цепочек с учетом типов документов"""
        prompts = {
            "legal": PromptTemplate(
                template="""Вы юридический ассистент. Анализируйте документы и отвечайте точно.
                            Контекст: {context}
                            Вопрос: {input}  # Изменено с question на input
                            Ответ должен содержать ссылки на конкретные статьи или разделы.""",
                input_variables=["context", "input"],  # Обновлено
            ),
            "qa": PromptTemplate(
                template="""Отвечайте кратко и по делу на вопросы пользователя.
                            Вопрос: {input}  # Изменено
                            Контекст: {context}
                            Если ответа нет в контексте, скажите "Информация не найдена".""",
                input_variables=["context", "input"],  # Обновлено
            ),
            "default": PromptTemplate(
                template="""Ответьте на вопрос на основе предоставленного контекста.
                            Контекст: {context}
                            Вопрос: {input}  # Изменено
                            Ответ:""",
                input_variables=["context", "input"],  # Обновлено
            ),
        }

        self.qa_chains = {}

        if not self.vector_db.db:
            raise RuntimeError("Vector database not initialized")

        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain

        available_types = self.get_available_doc_types() or ["default"]

        for doc_type in available_types:
            prompt = prompts.get(doc_type, prompts["default"])

            combine_chain = create_stuff_documents_chain(self.llm, prompt)

            retriever = self.vector_db.db.as_retriever(
                search_kwargs={
                    "k": 3,
                    "filter": {"document_type": doc_type},
                }
            )

            self.qa_chains[doc_type] = create_retrieval_chain(retriever, combine_chain)

    def close(self):
        """Корректное закрытие ресурсов"""
        if hasattr(self, "vector_db") and self.vector_db:
            self.vector_db.close()
        if hasattr(self, "llm"):
            # Для ChatOpenAI можно добавить очистку если нужно
            pass
        gc.collect()

    def query(
        self, question: str, doc_type: Optional[str] = None, use_cache: bool = False
    ) -> str:
        """Поиск с возможностью указания типа документа"""
        if not question or not isinstance(question, str):
            return "Вопрос должен быть непустой строкой"

        try:
            # Инициализация LLM при первом запросе (если не была инициализирована ранее)
            if self.llm is None:
                self.llm_provider = create_llm_provider(self.config.__dict__)
                self.llm = self.llm_provider.get_llm()
                self._init_chains()
                print(f"Инициализирована LLM: {self.llm_provider.get_model_name()}")

            # Принудительная проверка базы
            if not self.vector_db.db:
                self.vector_db.load_or_create()
            # Определяем тип документа
            target_doc_type = doc_type if doc_type in self.qa_chains else "default"

            # Проверяем наличие цепи
            if target_doc_type not in self.qa_chains:
                return (
                    f"Не найдена цепь обработки для типа документа: {target_doc_type}"
                )

            # Выполняем запрос
            result = self.qa_chains[target_doc_type].invoke(
                {"input": question}  # Ключ должен быть "input"
            )

            # Форматируем ответ
            answer = result.get("answer", "Ответ не найден")
            if isinstance(answer, str):
                return answer
            return str(answer)

        except Exception as e:
            print(f"Ошибка при выполнении запроса: {str(e)}")
            return f"Произошла ошибка: {str(e)}"


def clean_data(vector_db: Optional[VectorDatabase] = None):
    """Clean existing ChromaDB indexes."""
    if vector_db:
        vector_db.db = None
        vector_db.cache_db = None
        gc.collect()

    for path in [config.CHROMA_DB_PATH, config.CHROMA_CACHE_PATH]:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"Removed {path}")
            except PermissionError as e:
                print(f"Error removing {path}: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")


def clear_semantic_cache(vector_db: VectorDatabase):
    """Clear semantic cache completely."""
    if vector_db.cache_db is None:
        vector_db.load_or_create_cache()

    if (
        vector_db.cache_db
        and hasattr(vector_db.cache_db, "_collection")
        and vector_db.cache_db._collection.count() > 0
    ):
        all_ids = vector_db.cache_db._collection.get()["ids"]
        if all_ids:
            vector_db.cache_db.delete(ids=all_ids)

    vector_db.cache_db = None
    gc.collect()


def run_indexing(vector_db: VectorDatabase):
    """Run document indexing."""
    rag = RAGSystem(config)
    rag.vector_db = vector_db
    rag.initialize()
    rag.close()


def run_interactive_chat(vector_db: VectorDatabase):
    """Run interactive chat with proper resource cleanup"""
    rag = None
    try:
        rag = RAGSystem(config)
        rag.initialize()

        print("\n=== Интерактивный режим ===")
        print("Введите ваш вопрос или одну из команд:")
        print("- 'exit', 'quit', 'q' - выход")
        print("- 'menu' - вернуться в меню")

        while True:
            try:
                user_input = input("\nВаш запрос: ").strip()

                if user_input.lower() in ("exit", "quit", "q"):
                    print("Завершение сеанса...")
                    break
                elif user_input.lower() == "menu":
                    print("Возврат в главное меню...")
                    break

                if user_input:
                    answer = rag.query(user_input, use_cache=False)
                    print(f"\nОтвет: {answer}")

            except KeyboardInterrupt:
                print("\nПрервано пользователем")
                break
            except Exception as e:
                print(f"\nОшибка: {e}")

    finally:
        if rag:
            rag.close()
        print("Ресурсы освобождены.")


# Этот код перенесен в main.py
