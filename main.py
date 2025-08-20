from services.rag_system import (
    RAGSystem,
    VectorDatabase,
    clean_data,
    clear_semantic_cache,
    run_indexing,
    run_interactive_chat,
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


def main():
    vector_db = None
    try:
        vector_db = VectorDatabase(
            config.CHROMA_DB_PATH, config.CHROMA_CACHE_PATH, EmbeddingManager(config)
        )

        while True:
            display_menu()
            choice = input("\nВыберите вариант: ").strip()

            if not choice:  # Пустой ввод
                continue

            if choice == "1":
                clean_data(vector_db)
            elif choice == "2":
                run_indexing(vector_db)
            elif choice == "3":
                run_interactive_chat(vector_db)
            elif choice == "4":
                clear_semantic_cache(vector_db)
            elif choice == "0":
                print("\nЗавершение работы программы...")
                break
            else:
                print("Неверный выбор, попробуйте снова")

    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
    finally:
        if vector_db:
            vector_db.close()
        print("Программа завершена.")


if __name__ == "__main__":
    main()
