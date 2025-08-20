import json
import os
from typing import List
from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text
from unstructured.partition.auto import partition
from config import Config
from langchain_core.documents import Document
from utils.json_splitter import JsonTextSplitter


class DocumentParser:
    def __init__(self, config: Config):
        self.config = config

    def needs_ocr(self, file_path: str) -> bool:
        """Determine if PDF requires OCR processing."""
        relative_path = os.path.relpath(file_path, start=self.config.INPUT_DIR).lower()
        for keyword in self.config.PDF_PROCESSING["ocr_path_keywords"]:
            if keyword in relative_path:
                return True

        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                text = "".join(
                    page.extract_text() or ""
                    for page in reader.pages[: self.config.PDF_MAX_PAGES_CHECK]
                )

                if len(text.strip()) < self.config.PDF_MIN_TEXT_LENGTH or any(
                    kw in text.lower()
                    for kw in self.config.PDF_PROCESSING["ocr_keywords"]
                ):
                    return True
        except Exception:
            return True

        return False

    def parse_document(self, file_path: str) -> str:
        """Parse document content with automatic OCR detection for PDFs."""
        if file_path.lower().endswith(".rtf"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return rtf_to_text(f.read())
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="cp1251") as f:
                    return rtf_to_text(f.read())

        elif file_path.lower().endswith(".pdf"):
            try:
                strategy = (
                    self.config.PDF_PROCESSING["ocr_strategy"]
                    if self.needs_ocr(file_path)
                    else self.config.PDF_PROCESSING["default_strategy"]
                )

                if self.needs_ocr(file_path):
                    print(f"  Применение OCR к: {os.path.basename(file_path)}")

                elements = partition(
                    filename=file_path,
                    strategy=strategy,
                    languages=self.config.PDF_PROCESSING["ocr_languages"],
                    pdf_infer_table_structure=True,
                    max_pages=self.config.PDF_MAX_PAGES_PROCESS,
                    use_gpu=False,
                )

                text_content = "\n\n".join(
                    e.text for e in elements if e.text and e.text.strip()
                )

                if not text_content.strip():
                    raise ValueError("Распознанный текст пуст")
                return text_content

            except Exception as e:
                print(f"  Ошибка обработки PDF: {str(e)}")
                print("  Повторная попытка с альтернативными параметрами...")
                try:
                    elements = partition(
                        filename=file_path,
                        strategy="hi_res",
                        languages=["rus", "eng"],
                        pdf_infer_table_structure=False,
                        use_gpu=False,
                    )
                    return "\n\n".join(e.text for e in elements if e.text)
                except Exception as e2:
                    print(f"  Критическая ошибка обработки PDF: {str(e2)}")
                    return ""

        elif file_path.lower().endswith((".docx", ".doc", ".txt", ".md")):
            elements = partition(filename=file_path)
            return "\n\n".join(e.text for e in elements if e.text and e.text.strip())

        elif file_path.lower().endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            splitter = JsonTextSplitter()
            return splitter.split_json(data)

        raise ValueError(f"Unsupported file format: {file_path}")
