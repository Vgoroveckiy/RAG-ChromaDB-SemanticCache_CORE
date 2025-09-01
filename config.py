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

        # Модель для эмбеддинга
        self.EMBEDDING_MODEL = "ai-forever/sbert_large_nlu_ru"
        self.LLM_TEMPERATURE = 0.5

        # Настройки провайдеров LLM
        self.LLM_PROVIDER = "sber"  # openai | yandex | sber | openrouter

        # Модели для провайдеров LLM (обязательные)
        self.OPENAI_MODEL_ID = "gpt-4o-mini"  # Модель для OpenAI
        self.YANDEX_MODEL_ID = "general"  # Модель для Yandex
        self.SBER_MODEL_ID = "GigaChat"  # Модель для Sber

        # Платные модели для OpenRouter
        # self.OPENROUTER_MODEL_ID = "openai/gpt-oss-20b"  # Модель для OpenRouter
        # self.OPENROUTER_MODEL_ID = (
        #     "deepseek/deepseek-chat-v3-0324"  # Модель для OpenRouter
        # )

        # Бесплатные модели для OpenRouter
        # self.OPENROUTER_MODEL_ID = "openai/gpt-oss-20b:free"  # Модель для OpenRouter
        self.OPENROUTER_MODEL_ID = (
            "deepseek/deepseek-chat-v3-0324:free"  # Модель для OpenRouter
        )

        # API ключи теперь загружаются из переменных окружения
        self.YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")  # API-ключ для Yandex
        self.YANDEX_IAM_TOKEN = os.getenv(
            "YANDEX_IAM_TOKEN", ""
        )  # IAM-токен для Yandex Cloud
        self.SBER_API_KEY = os.getenv("SBER_API_KEY", "")  # API-ключ для Sber LLM
        self.OPENROUTER_API_KEY = os.getenv(
            "OPENROUTER_API_KEY", ""
        )  # API key for OpenRouter

        # Провайдеры чата
        self.CHAT_PROVIDERS = {
            "console": {
                "enabled": True,
                "class": "providers.console_provider.ConsoleProvider",
            },
            "telegram": {
                "enabled": True,
                "class": "providers.telegram_provider.TelegramProvider",
                "params": {"token": os.getenv("TELEGRAM_TOKEN", "")},
            },
        }

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
        if not os.getenv("YANDEX_API_KEY") and not os.getenv("YANDEX_IAM_TOKEN"):
            raise ValueError(
                "Either YANDEX_API_KEY or YANDEX_IAM_TOKEN environment variable must be set in .env"
            )

    elif provider == "sber":
        if not config.SBER_MODEL_ID:
            raise ValueError("SBER_MODEL_ID must be set for Sber provider")
        if not os.getenv("SBER_API_KEY"):
            raise ValueError("SBER_API_KEY environment variable must be set in .env")

    elif provider == "openrouter":
        if not config.OPENROUTER_MODEL_ID:
            raise ValueError("OPENROUTER_MODEL_ID must be set for OpenRouter provider")
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError(
                "OPENROUTER_API_KEY environment variable must be set in .env"
            )

    else:
        raise ValueError(f"Unsupported provider: {provider}")


# Проверка конфигурации
validate_config(config)

# Проверка API ключей теперь выполняется в validate_config()

# Создание директорий
os.makedirs(config.INPUT_DIR, exist_ok=True)
os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
os.makedirs(config.CHROMA_CACHE_PATH, exist_ok=True)
print(
    f"Директории проверены: {config.INPUT_DIR}, {config.CHROMA_DB_PATH}, {config.CHROMA_CACHE_PATH}"
)
