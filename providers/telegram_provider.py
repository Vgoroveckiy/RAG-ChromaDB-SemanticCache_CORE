import asyncio
import logging
import threading
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from core.chat_provider import ChatProvider


class TelegramProvider(ChatProvider):
    """Провайдер для взаимодействия через Telegram"""

    def __init__(self, token: str):
        self.token = token
        self.bot = Bot(token=token)
        self.dp = Dispatcher()
        self.message_handler = None
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.thread = None

    def start(self):
        """Запускает Telegram бота в отдельном потоке"""
        if self.running:
            return

        self.running = True
        # Создаем поток как демон, чтобы он не блокировал завершение программы
        self.thread = threading.Thread(target=self._run_bot, daemon=True)
        self.thread.start()

    def _run_bot(self):
        """Основной цикл работы Telegram бота в отдельном потоке"""
        self.logger.info("Запуск Telegram бота")

        # Создаем новый цикл событий для этого потока
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Регистрируем обработчики
        self.dp.message.register(self._handle_message)
        self.dp.message(Command("start"))(self._handle_start)
        self.dp.message(Command("menu"))(self._handle_menu)
        self.dp.message(Command("help"))(
            self._handle_start
        )  # Добавляем help как аналог start

        try:
            loop.run_until_complete(
                self.dp.start_polling(self.bot, handle_signals=False)
            )
        except Exception as e:
            self.logger.error(f"Ошибка в Telegram боте: {e}")
        finally:
            self.running = False

    async def _handle_start(self, message: types.Message):
        """Обработчик команды /start"""
        await message.answer(
            "Добро пожаловать в чат-бот RAG системы!\nОтправьте ваш запрос."
        )

    async def _handle_menu(self, message: types.Message):
        """Обработчик команды /menu"""
        if self.message_handler:
            response = self.message_handler("menu")
            await message.answer(response)

    async def _handle_message(self, message: types.Message):
        """Обработчик входящих сообщений"""
        if not message.text or not self.message_handler:
            return

        # Обрабатываем текст сообщения
        response = self.message_handler(message.text)
        await message.answer(response)

    def send_message(self, message: str):
        """Отправляет сообщение через Telegram (не реализовано, так как требует chat_id)"""
        # Этот метод будет реализован позже при необходимости
        self.logger.warning(
            "Метод send_message не реализован для группового использования"
        )

    def register_message_handler(self, handler: callable):
        """Регистрирует обработчик входящих сообщений"""
        self.message_handler = handler

    def stop(self):
        """Останавливает Telegram бота (демонический поток завершится автоматически)"""
        self.running = False
        self.logger.info("Telegram бот получил сигнал остановки")
        # Явное завершение не требуется, так как поток демонический
