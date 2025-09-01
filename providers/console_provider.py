from core.chat_provider import ChatProvider


class ConsoleProvider(ChatProvider):
    """Провайдер для консольного взаимодействия"""

    def __init__(self):
        self.message_handler = None

    def start(self):
        """Заглушка для метода запуска (не требуется для консоли)"""
        pass

    def stop(self):
        """Заглушка для метода остановки (не требуется для консоли)"""
        pass

    def send_message(self, message: str):
        """Выводит сообщение в консоль"""
        print(f"\nОтвет: {message}")

    def run_in_foreground(self):
        """Запускает консольный интерфейс в основном потоке"""
        print("\n=== Консольный чат ===")
        print("Введите ваш вопрос или команду:")
        print("- 'exit', 'quit', 'q' - выход")
        print("- 'menu' - вернуться в меню")

        while True:
            try:
                user_input = input("\nВаш запрос: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "q", "menu"):
                    print("Возврат в главное меню...")
                    return

                if self.message_handler:
                    response = self.message_handler(user_input)
                    self.send_message(response)

            except KeyboardInterrupt:
                print("\nКонсольный ввод прерван")
                return
            except Exception as e:
                print(f"\nОшибка в консольном провайдере: {e}")

    def register_message_handler(self, handler: callable):
        """Регистрирует обработчик входящих сообщений"""
        self.message_handler = handler
