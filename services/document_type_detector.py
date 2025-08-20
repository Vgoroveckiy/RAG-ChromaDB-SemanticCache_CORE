from typing import Literal, Tuple
import os
import re
from config import Config

DocumentType = Literal["legal", "qa", "default"]


class DocumentTypeDetector:
    def __init__(self, config: Config):
        self.config = config
        self.folder_to_type = {
            folder.lower(): doc_type
            for doc_type, cfg in config.DOCUMENT_TYPE_CONFIG.items()
            for folder in cfg.get("path_keywords", [])
        }
        self.prefix_to_type = {
            prefix.lower(): doc_type
            for doc_type, cfg in config.DOCUMENT_TYPE_CONFIG.items()
            for prefix in cfg.get("filename_prefixes", [])
        }

    def detect(self, file_path: str, content: str = "") -> Tuple[DocumentType, str]:
        relative_path = os.path.relpath(file_path, start=self.config.INPUT_DIR)
        path_parts = [p.lower() for p in relative_path.split(os.sep)]
        filename = path_parts[-1]

        # 1. Check by folder
        for folder in path_parts[:-1]:
            if folder in self.folder_to_type:
                return self.folder_to_type[folder], f"по каталогу '{folder}'"

        # 2. Check by filename prefix
        for prefix, doc_type in self.prefix_to_type.items():
            if filename.startswith(prefix):
                return doc_type, f"по префиксу '{prefix}'"

        # 3. Content analysis
        content_lower = content.lower()
        for doc_type, type_config in self.config.DOCUMENT_TYPE_CONFIG.items():
            for keyword in type_config.get("content_keywords", []):
                if keyword.lower() in content_lower:
                    return doc_type, f"по ключевому слову '{keyword}'"
            for pattern in type_config.get("content_patterns", []):
                if re.search(pattern, content, re.IGNORECASE):
                    return doc_type, f"по шаблону '{pattern}'"

        return "default", "автоматически (не удалось определить)"
