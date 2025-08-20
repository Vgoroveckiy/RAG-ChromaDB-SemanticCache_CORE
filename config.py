import os

from dotenv import load_dotenv

load_dotenv()

# Отключаем GPU для обработки PDF
os.environ["UNSTRUCTURED_DISABLE_GPU"] = "1"


class Config:
    def __init__(self):
        # Пути и директории
        self.INPUT_DIR = "data"
        self.CHROMA_DB_PATH = "./chroma_db"
        self.CHROMA_CACHE_PATH = "./chroma_cache"

        # Модели
        self.EMBEDDING_MODEL = "ai-forever/sbert_large_nlu_ru"
        self.LLM_TEMPERATURE = 0.5

        # Настройки провайдеров LLM
        self.LLM_PROVIDER = "openai"  # openai | yandex | sber | другие

        # Модели для провайдеров (обязательные)
        self.OPENAI_MODEL_ID = "gpt-4o-mini"  # Модель для OpenAI
        self.YANDEX_MODEL_ID = "general"  # Модель для Yandex
        self.SBER_MODEL_ID = "sber-large"  # Модель для Sber

        # API ключи
        self.YANDEX_API_KEY = ""  # API-ключ для Yandex
        self.YANDEX_IAM_TOKEN = ""  # IAM-токен для Yandex Cloud
        self.SBER_API_KEY = ""  # API-ключ для Sber LLM

        # Настройки обработки документов
        self.SUPPORTED_EXTENSIONS = (
            ".pdf",
            ".doc",
            ".docx",
            ".txt",
            ".json",
            ".rtf",
            ".md",
        )
        self.CACHE_TTL_DAYS = 30
        self.DOMAIN_SPECIALTY = "юридические вопросы"

        # Новые параметры
        self.MTIME_TOLERANCE_SECONDS = 300  # Допуск для времени модификации файлов
        self.CHROMA_BATCH_SIZE = 1000  # Размер батча для пагинации в ChromaDB
        self.PDF_MAX_PAGES_CHECK = 3  # Количество страниц для проверки OCR
        self.PDF_MIN_TEXT_LENGTH = 100  # Минимальная длина текста для определения OCR
        self.CACHE_SIMILARITY_THRESHOLD = 0.1  # Порог схожести для семантического кэша
        self.RETRIEVER_K_LEGAL = 4  # Количество возвращаемых документов для legal
        self.RETRIEVER_K_DEFAULT = (
            3  # Количество возвращаемых документов для других типов
        )
        self.RETRIEVER_SCORE_THRESHOLD_LEGAL = 0.75  # Порог для retriever (legal)
        self.RETRIEVER_SCORE_THRESHOLD_DEFAULT = 0.65  # Порог для retriever (другие)

        self.PDF_MAX_PAGES_PROCESS = (
            100  # Максимальное количество страниц для обработки PDF
        )

        self.PDF_PROCESSING = {
            "default_strategy": "fast",
            "ocr_strategy": "hi_res",
            "ocr_languages": ["rus", "eng"],
            "ocr_keywords": ["сканирован", "копия", "image"],
            "ocr_path_keywords": ["scans/", "ocr/"],
            "use_gpu": False,  # Отключаем GPU для обработки PDF
        }
        self.DOCUMENT_TYPE_CONFIG = {
            "legal": {
                "filename_prefixes": ["legal_", "contract_", "law_"],
                "path_keywords": ["legal", "contracts"],
                "content_keywords": ["договор", "сторона", "статья", "юрист"],
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "model": "ai-forever/sbert_large_nlu_ru",
                "separators": ["\n\nСТАТЬЯ", "\n\nРАЗДЕЛ", "\n\n", "\n"],
            },
            "qa": {
                "filename_prefixes": ["faq_", "qa_", "question_"],
                "path_keywords": ["faq", "questions"],
                "content_patterns": [
                    r"вопрос:\s*.+\s*ответ:\s*.+",
                    r"q:\s*.+\s*a:\s*.+",
                ],
                "chunk_size": 300,
                "chunk_overlap": 50,
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "separators": ["\n\nQ:", "\nA:", "\n\n", "\n"],
            },
            "json": {
                "filename_prefixes": ["data_", "json_"],
                "path_keywords": ["json_data", "json_files"],
                "content_keywords": ["json"],
                "chunk_size": 500,
                "chunk_overlap": 50,
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "separators": ["}", ","],
            },
        }


# Инициализация конфига
config = Config()


def validate_config(config):
    """Проверка обязательных параметров конфигурации"""
    provider = config.LLM_PROVIDER

    if provider == "openai":
        if not config.OPENAI_MODEL_ID:
            raise ValueError("OPENAI_MODEL_ID must be set for OpenAI provider")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable must be set in .env")

    elif provider == "yandex":
        if not config.YANDEX_MODEL_ID:
            raise ValueError("YANDEX_MODEL_ID must be set for Yandex provider")
        if not config.YANDEX_API_KEY and not config.YANDEX_IAM_TOKEN:
            raise ValueError("Either YANDEX_API_KEY or YANDEX_IAM_TOKEN must be set")

    elif provider == "sber":
        if not config.SBER_MODEL_ID:
            raise ValueError("SBER_MODEL_ID must be set for Sber provider")
        if not config.SBER_API_KEY:
            raise ValueError("SBER_API_KEY must be set for Sber provider")

    else:
        raise ValueError(f"Unsupported provider: {provider}")


# Проверка конфигурации
validate_config(config)

# Проверка API ключа OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")
os.environ["OPENAI_API_KEY"] = api_key

# Создание директорий
os.makedirs(config.INPUT_DIR, exist_ok=True)
os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
os.makedirs(config.CHROMA_CACHE_PATH, exist_ok=True)
print(
    f"Директории проверены: {config.INPUT_DIR}, {config.CHROMA_DB_PATH}, {config.CHROMA_CACHE_PATH}"
)
