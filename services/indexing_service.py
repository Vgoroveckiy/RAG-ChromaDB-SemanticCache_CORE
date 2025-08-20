import os
import json
import gc
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from managers.vector_db_manager import VectorDatabase
from managers.embedding_manager import EmbeddingManager
from utils import chunk_utils
from config import config


def update_document_in_chroma(
    vector_db: VectorDatabase,
    file_path: str,
    full_text_content: str,
    metadata: dict,
    current_file_hash: str,
    current_last_modified: float,
    stored_doc_id: Optional[str] = None,
) -> Tuple[List[Document], List[str]]:
    """Update document in ChromaDB."""
    metadata.update(
        {
            "file_hash_full": current_file_hash,
            "last_modified": current_last_modified,
            "processing_time": datetime.now().isoformat(),
        }
    )

    new_chunks = vector_db.embedding_manager.process_document(file_path)
    if not new_chunks:
        print(f"⚠️ No chunks generated for {file_path}")
        return [], []

    texts = [chunk.page_content for chunk in new_chunks]
    metadatas = [chunk.metadata for chunk in new_chunks]
    ids = [str(uuid.uuid4()) for _ in new_chunks]

    if stored_doc_id:
        vector_db.delete_documents([stored_doc_id])

    try:
        if not vector_db.db:
            vector_db.load_or_create()

        vector_db.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"✅ Added {len(ids)} chunks from {os.path.basename(file_path)}")
        return new_chunks, ids
    except Exception as e:
        print(f"❌ Failed to add texts to ChromaDB: {str(e)}")
        raise


def process_catalog_data(
    file_path: str, vector_db: VectorDatabase
) -> Tuple[List[Document], int]:
    """Process JSON catalog data"""
    documents = []
    added_count = 0
    base_filename = os.path.basename(file_path)
    print(f"\nОбработка каталога: {base_filename}")

    try:
        file_hash = vector_db.embedding_manager.get_file_hash(file_path)
        last_modified = os.path.getmtime(file_path)
        file_unchanged = False

        existing_docs = vector_db.get_documents_by_hash(file_hash)
        if (
            existing_docs
            and existing_docs.get("metadatas")
            and len(existing_docs["metadatas"]) > 0
        ):
            first_meta = existing_docs["metadatas"][0]
            if (
                isinstance(first_meta, dict)
                and first_meta.get("file_hash_full") == file_hash
                and abs(float(first_meta.get("last_modified", 0)) - last_modified)
                < config.MTIME_TOLERANCE_SECONDS
            ):
                file_unchanged = True
                documents.extend(
                    [
                        Document(
                            page_content=existing_docs["documents"][i],
                            metadata=existing_docs["metadatas"][i],
                        )
                        for i in range(len(existing_docs["ids"]))
                    ]
                )
                print("  Каталог не изменился, используется существующая индексация")
                return documents, 0

        if not file_unchanged:
            vector_db.delete_cached_entries_by_source(base_filename)
            print("  Обновление каталога...")
            chunks = chunk_utils.process_json_file(file_path)

            for chunk in chunks:
                item_path = chunk.metadata["file_path"]
                if vector_db.db:
                    vector_db.db.delete(where={"file_path": item_path})
                if vector_db.db:
                    vector_db.db.add_texts(
                        texts=[chunk.page_content],
                        metadatas=[chunk.metadata],
                        ids=[str(uuid.uuid4())],
                    )
                    added_count += 1
                documents.append(chunk)
            print(f"✅ Добавлено чанков: {len(chunks)}")
        return documents, added_count
    except Exception as e:
        print(f"Ошибка обработки каталога {file_path}: {e}")
        return [], 0
    finally:
        gc.collect()


def get_all_metadata(
    vector_db: VectorDatabase, batch_size: int = config.CHROMA_BATCH_SIZE
) -> List[Dict]:
    """Retrieve all metadata from ChromaDB with pagination."""
    return vector_db.get_all_metadata(batch_size)


def parse_files(directory: str, vector_db: VectorDatabase) -> List[Document]:
    """Parse files from directory and index them in ChromaDB."""
    documents = []
    print("\n=== Начало индексации документов ===")
    total_files = 0
    total_chunks = 0

    existing_files_meta = {}
    if vector_db.db and hasattr(vector_db.db, "_collection"):
        for meta in get_all_metadata(vector_db, config.CHROMA_BATCH_SIZE):
            if isinstance(meta, dict) and "file_path" in meta:
                rel_path = (
                    os.path.relpath(meta["file_path"], directory)
                    .replace("\\", "/")
                    .lower()
                )
                existing_files_meta[rel_path] = meta

    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.normpath(os.path.join(root, filename))
            rel_path = os.path.relpath(file_path, directory).replace("\\", "/").lower()
            total_files += 1
            print(f"\n[{total_files}] Обработка файла: {rel_path}")

            if not filename.lower().endswith(config.SUPPORTED_EXTENSIONS):
                print("  Пропуск: неподдерживаемый формат")
                continue

            try:
                current_hash = vector_db.embedding_manager.get_file_hash(file_path)
                current_mtime = float(os.path.getmtime(file_path))
                needs_reindex = True

                if rel_path in existing_files_meta:
                    existing_meta = existing_files_meta[rel_path]
                    if (
                        existing_meta.get("file_hash") == current_hash
                        and abs(
                            float(existing_meta.get("last_modified", 0)) - current_mtime
                        )
                        < config.MTIME_TOLERANCE_SECONDS
                    ):
                        needs_reindex = False
                        print(
                            "  Файл не изменился, используется существующая индексация"
                        )
                        continue

                if needs_reindex:
                    print("  Обновление индексации файла...")

                    if filename.lower().endswith(".json"):
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            if isinstance(data, list) and all(
                                isinstance(i, dict) for i in data
                            ):
                                print("  Обнаружен JSON-каталог, специальная обработка")
                                chunks, added_chunks = process_catalog_data(
                                    file_path, vector_db
                                )
                                total_chunks += added_chunks
                                continue
                        except Exception as e:
                            print(f"  Ошибка при проверке JSON: {e}")

                    content = vector_db.embedding_manager.parse_document(file_path)
                    doc_type, detection_reason = (
                        vector_db.embedding_manager.type_detector.detect(
                            file_path, content
                        )
                    )
                    metadata = chunk_utils.generate_metadata(
                        file_path, doc_type, current_hash, current_mtime
                    )
                    metadata["detection_reason"] = detection_reason

                    if rel_path in existing_files_meta:
                        vector_db.db.delete(where={"file_path": file_path})

                    chunks, ids = update_document_in_chroma(
                        vector_db,
                        file_path,
                        content,
                        metadata,
                        current_hash,
                        current_mtime,
                    )

                    chunk_count = len(chunks)
                    total_chunks += chunk_count
                    print(
                        f"  Добавлено чанков: {chunk_count} (тип: {doc_type}, причина: {detection_reason})"
                    )

            except Exception as e:
                print(f"  Ошибка обработки файла: {str(e)}")
                continue

    print("\n=== Итоги индексации ===")
    print(f"Всего обработано файлов: {total_files}")
    print(f"Всего добавлено чанков: {total_chunks}")
    print(
        f"Общее количество документов в базе: {vector_db.db._collection.count() if vector_db.db else 0}"
    )

    cleanup_deleted_files(vector_db, directory)
    return documents


def cleanup_deleted_files(vector_db: VectorDatabase, input_dir: str) -> None:
    """Check for deleted files and remove their entries from ChromaDB."""
    if not vector_db.db:
        return

    try:
        abs_input_dir = os.path.abspath(input_dir)
        files_to_delete_set = set()
        total_checked = 0

        for metadata in get_all_metadata(vector_db, config.CHROMA_BATCH_SIZE):
            if isinstance(metadata, dict):
                file_path = metadata.get("file_path", "")
                total_checked += 1

                if "#" in file_path:
                    main_file = file_path.split("#")[0]
                    if not os.path.exists(main_file):
                        files_to_delete_set.add(file_path)
                else:
                    if not os.path.exists(file_path):
                        files_to_delete_set.add(file_path)

        files_to_delete = list(files_to_delete_set)
        # Группируем удаленные файлы по основному файлу
        main_file_to_chunk_count = {}
        main_file_to_ids = {}

        for file_path in files_to_delete:
            if "#" in file_path:
                main_file = file_path.split("#")[0]
            else:
                main_file = file_path

            if main_file not in main_file_to_ids:
                main_file_to_ids[main_file] = []
                main_file_to_chunk_count[main_file] = 0

            # Получаем ID чанков для этого пути
            docs = vector_db.db.get(where={"file_path": file_path})
            if docs and docs.get("ids"):
                ids = docs["ids"]
                main_file_to_ids[main_file].extend(ids)
                main_file_to_chunk_count[main_file] += len(ids)

        # Удаляем и выводим результаты по основным файлам
        total_deleted = 0
        for main_file, ids in main_file_to_ids.items():
            if not ids:
                continue

            count = main_file_to_chunk_count[main_file]
            total_deleted += count
            vector_db.delete_documents(ids)
            vector_db.delete_cached_entries_by_source(os.path.basename(main_file))
            print(f"Удален файл: {main_file} | Удалено чанков: {count}")

        if total_deleted > 0:
            print(f"Всего удалено чанков: {total_deleted}")
        else:
            print("Нет чанков для удаления")
    except Exception as e:
        print(f"Ошибка при очистке удаленных файлов: {e}")
