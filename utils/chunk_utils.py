import json
import hashlib
import os
from langchain_core.documents import Document
from typing import List, Dict


def create_chunks(content: str, metadata: dict, doc_type: str) -> List[Document]:
    """Создает чанки из контента документа с метаданными"""
    # В реальной реализации здесь будет логика разделения на чанки
    # Для примера создаем один чанк со всем содержимым
    return [Document(page_content=content, metadata=metadata)]


def process_json_file(file_path: str) -> List[Document]:
    """Обрабатывает JSON-файл и создает чанки для каждого элемента"""
    base_filename = os.path.basename(file_path)
    file_hash = hashlib.sha256(open(file_path, "rb").read()).hexdigest()
    last_modified = os.path.getmtime(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)

    chunks = []
    for i, item in enumerate(catalog_data):
        item_path = f"{file_path}#{i}"

        item_content = json.dumps(item, ensure_ascii=False)

        metadata = {
            "source": base_filename,
            "item_name": item.get("name", "N/A"),
            "item_url": item.get("url", "N/A"),
            "index_in_catalog": i,
            "file_path": item_path,
            "file_hash": hashlib.sha256(item_content.encode()).hexdigest(),
            "file_hash_full": file_hash,
            "last_modified": last_modified,
            "document_type": "qa",
        }
        chunks.append(Document(page_content=item_content, metadata=metadata))

    return chunks


def generate_metadata(
    file_path: str, doc_type: str, file_hash: str, last_modified: float
) -> Dict:
    """Генерирует стандартные метаданные для документа"""
    return {
        "source": os.path.basename(file_path),
        "file_path": file_path,
        "file_hash": file_hash,
        "last_modified": last_modified,
        "document_type": doc_type,
    }
