from services.rag_system import (
    RAGSystem,
    VectorDatabase,
    clean_data,
    clear_semantic_cache,
    run_indexing,
)
from config import config
from managers.embedding_manager import EmbeddingManager


def display_menu():
    """Display main menu."""
    print("\n" + "=" * 50)
    print("1. Очистить данные (индексы ChromaDB)")
    print("2. Индексировать документы")
    print("3. Интерактивный чат")
    print("4. Очистить кеш семантики")
    print("0. Выход")
    print("=" * 50)


from managers.provider_manager import ProviderManager


def main():
    vector_db = None
    provider_manager = None

    try:
        vector_db = VectorDatabase(
            config.CHROMA_DB_PATH, config.CHROMA_CACHE_PATH, EmbeddingManager(config)
        )

        # Инициализация менеджера провайдеров (автоматически запускает провайдеры)
        provider_manager = ProviderManager(vector_db, config)

        while True:
            display_menu()
            print()  # Пустая строка для визуального разделения
            choice = input("\nВыберите вариант: ").strip()

            if not choice:  # Пустой ввод
                continue

            if choice == "1":
                clean_data(vector_db)
            elif choice == "2":
                run_indexing(vector_db)
            elif choice == "3":
                if "console" in provider_manager.providers:
                    # Запускаем консольный чат напрямую
                    provider_manager.providers["console"].run_in_foreground()
                else:
                    print("Консольный провайдер не доступен")
            elif choice == "4":
                clear_semantic_cache(vector_db)
            elif choice == "0":
                print("\nЗавершение работы программы...")
                break
            else:
                print(f"Неверный выбор: '{choice}', попробуйте снова")

    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
    finally:
        # Явно останавливаем менеджер провайдеров перед выходом
        if provider_manager:
            provider_manager.stop_all()
        # Закрываем векторную базу данных
        if vector_db:
            vector_db.close()
        print("Программа завершена.")
        exit(0)  # Гарантированное завершение работы


if __name__ == "__main__":
    main()
