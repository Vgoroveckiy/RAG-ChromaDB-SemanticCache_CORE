import importlib
import logging
from typing import Dict, Callable
from core.chat_provider import ChatProvider
from services.rag_system import RAGSystem
from config import Config
from managers.vector_db_manager import VectorDatabase


class ProviderManager:
    """Менеджер для управления провайдерами чата"""

    def __init__(self, vector_db: VectorDatabase, config: Config):
        self.providers: Dict[str, ChatProvider] = {}
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.vector_db = vector_db
        # Создаем RAGSystem для обработки запросов
        self.rag_system = RAGSystem(config)
        self.rag_system.vector_db = vector_db
        self.rag_system.load_for_query()
        self.start_providers()  # Загружаем и запускаем провайдеры

        # Регистрируем обработчик сообщений
        self.register_message_handler(self._handle_message)

    def start_providers(self):
        """Загружает и запускает всех провайдеров из конфигурации"""
        if not hasattr(self.config, "CHAT_PROVIDERS"):
            self.logger.warning("В конфигурации отсутствует раздел CHAT_PROVIDERS")
            return

        for name, provider_config in self.config.CHAT_PROVIDERS.items():
            if not provider_config.get("enabled", False):
                continue

            try:
                # Динамическая загрузка класса провайдера
                module_path, class_name = provider_config["class"].rsplit(".", 1)
                module = importlib.import_module(module_path)
                provider_class = getattr(module, class_name)

                # Создание экземпляра провайдера
                params = provider_config.get("params", {})
                provider = provider_class(**params)
                self.providers[name] = provider
                self.logger.info(f"Провайдер '{name}' загружен")

                # Запуск провайдера (кроме консольного)
                if name != "console":
                    provider.start()
                    self.logger.info(f"Провайдер '{name}' запущен")

            except Exception as e:
                self.logger.error(f"Ошибка инициализации провайдера '{name}': {e}")

    def stop_all(self):
        """Останавливает все провайдеры, кроме консольного"""
        for name, provider in self.providers.items():
            try:
                # Консольный провайдер не требует остановки
                if name != "console":
                    provider.stop()
                    self.logger.info(f"Провайдер '{name}' остановлен")
            except Exception as e:
                self.logger.error(f"Ошибка остановки провайдера '{name}': {e}")

    def register_message_handler(self, handler: Callable[[str], str]):
        """Регистрирует обработчик входящих сообщений для всех провайдеров"""
        for provider in self.providers.values():
            provider.register_message_handler(handler)

    def _handle_message(self, message: str) -> str:
        """Обработчик входящих сообщений для всех провайдеров"""
        if message.lower() == "menu":
            return "Возврат в главное меню..."
        return self.rag_system.query(message)

    def send_to_provider(self, provider_name: str, message: str):
        """Отправляет сообщение через указанный провайдер"""
        if provider_name in self.providers:
            try:
                self.providers[provider_name].send_message(message)
            except Exception as e:
                self.logger.error(
                    f"Ошибка отправки через провайдер '{provider_name}': {e}"
                )
        else:
            self.logger.warning(f"Провайдер '{provider_name}' не найден")
