from abc import ABC, abstractmethod


class ChatProvider(ABC):
    """Абстрактный базовый класс для провайдеров чата"""

    @abstractmethod
    def start(self):
        """Запускает провайдер для приема сообщений"""
        pass

    @abstractmethod
    def send_message(self, message: str):
        """Отправляет сообщение пользователю через провайдера"""
        pass

    @abstractmethod
    def register_message_handler(self, handler: callable):
        """Регистрирует обработчик входящих сообщений

        Args:
            handler: Функция обработки сообщений вида handler(message: str) -> str
        """
        pass

    @abstractmethod
    def stop(self):
        """Останавливает провайдер и освобождает ресурсы"""
        pass
