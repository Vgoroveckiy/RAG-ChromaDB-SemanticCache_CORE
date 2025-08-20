import sys
import os
import json
from pathlib import Path

# Добавляем корень проекта в sys.path для корректного импорта
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from managers.embedding_manager import EmbeddingManager
from utils.text_cleaner import LegalTextCleaner
from langchain_core.documents import Document  # Импорт класса Document
import utils.chunk_utils as chunk_utils  # Импорт нового модуля


def analyze_document(file_path, clean_text=True):
    """Анализирует чанкинг документа и генерирует Markdown-отчет"""
    # Инициализация менеджера эмбеддингов
    manager = EmbeddingManager(config)

    # Инициализация очистителя текста
    cleaner = LegalTextCleaner() if clean_text else None

    # Специальная обработка для JSON
    if file_path.lower().endswith(".json"):
        # Используем новый модуль для обработки JSON
        chunks = chunk_utils.process_json_file(file_path)

        # Формирование отчета
        report = f"# Отчет по чанкингу JSON-документа\n\n"
        report += f"**Файл:** `{file_path}`\n\n"
        report += f"**Всего элементов:** {len(chunks)}\n\n"

        # Сохранение информации о типах элементов
        type_counts = {}
        for chunk in chunks:
            try:
                item = json.loads(chunk.page_content)
                item_type = "object"
                if "url" in item and "name" in item:
                    item_type = "catalog_item"
                elif "question" in item and "answer" in item:
                    item_type = "qa_item"
                type_counts[item_type] = type_counts.get(item_type, 0) + 1
            except json.JSONDecodeError:
                continue

        report += "**Типы элементов:**\n"
        for item_type, count in type_counts.items():
            report += f"- {item_type}: {count} элементов\n"
        report += "\n"

        # Добавляем флаг очистки текста в метаданные
        for chunk in chunks:
            chunk.metadata["text_cleaned"] = str(clean_text)
    else:
        # Обработка обычных документов
        raw_text = manager.parse_document(file_path)
        # Очистка текста (если включена)
        processed_text = cleaner.clean(raw_text) if clean_text else raw_text

        # Определение типа документа
        doc_type, reason = manager.type_detector.detect(file_path, processed_text)

        # Используем новый модуль для генерации метаданных
        metadata = chunk_utils.generate_metadata(
            file_path,
            doc_type,
            manager.get_file_hash(file_path),
            os.path.getmtime(file_path),
        )
        metadata["detection_reason"] = reason
        metadata["text_cleaned"] = str(clean_text)

        # Создание чанков из обработанного текста
        chunks = manager.splitters[doc_type].create_documents(
            [processed_text], [metadata]
        )

        # Формирование отчета
        report = f"# Отчет по чанкингу документа\n\n"
        report += f"**Файл:** `{file_path}`\n"
        report += f"**Очистка текста:** {'Да' if clean_text else 'Нет'}\n\n"
        report += f"**Всего чанков:** {len(chunks)}\n\n"

        # Сбор информации о позициях для расчета перекрытия
        chunk_positions = []
        for chunk in chunks:
            # Проверка наличия start_index в метаданных
            if "start_index" not in chunk.metadata:
                print(
                    f"Предупреждение: для чанка отсутствует start_index. Используется 0."
                )
            start_index = chunk.metadata.get("start_index", 0)
            end_index = start_index + len(chunk.page_content)
            chunk_positions.append((start_index, end_index))

        # Расчет перекрытий между последовательными чанками
        overlaps = []
        for i in range(1, len(chunk_positions)):
            prev_end = chunk_positions[i - 1][1]
            curr_start = chunk_positions[i][0]
            overlap = max(0, prev_end - curr_start)
            overlaps.append(overlap)

        # Общая статистика по перекрытию
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        report += f"**Среднее перекрытие:** {avg_overlap:.1f} символов\n\n"

    # Общая часть для всех типов документов

    # Сбор информации о позициях для расчета перекрытия
    chunk_positions = []
    # Для JSON пропускаем расчет позиций и перекрытий
    if not file_path.lower().endswith(".json"):
        chunk_positions = []
        for chunk in chunks:
            # Проверка наличия start_index в метаданных
            if "start_index" not in chunk.metadata:
                print(
                    f"Предупреждение: для чанка отсутствует start_index. Используется 0."
                )
            start_index = chunk.metadata.get("start_index", 0)
            end_index = start_index + len(chunk.page_content)
            chunk_positions.append((start_index, end_index))

        # Расчет перекрытий между последовательными чанками
        overlaps = []
        for i in range(1, len(chunk_positions)):
            prev_end = chunk_positions[i - 1][1]
            curr_start = chunk_positions[i][0]
            overlap = max(0, prev_end - curr_start)
            overlaps.append(overlap)

        # Общая статистика по перекрытию
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        report += f"**Среднее перекрытие:** {avg_overlap:.1f} символов\n\n"

    for i, chunk in enumerate(chunks, 1):
        # Получение метаданных
        metadata_str = "\n".join(
            [f"- **{key}:** {value}" for key, value in chunk.metadata.items()]
        )

        # Для JSON добавляем специальную информацию
        if file_path.lower().endswith(".json"):
            report += f"## Элемент {i}\n\n"
            report += f"**Метаданные:**\n{metadata_str}\n\n"
            # Форматирование JSON с отступами
            try:
                json_content = json.loads(chunk.page_content)
                formatted_json = json.dumps(json_content, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                formatted_json = chunk.page_content
            report += f"**Содержимое:**\n```json\n{formatted_json}\n```\n\n"
        else:
            # Получение позиции чанка
            start_index = chunk.metadata.get("start_index", 0)
            end_index = start_index + len(chunk.page_content)
            position_info = f"{start_index}-{end_index}"

            # Форматирование текста чанка
            chunk_text = chunk.page_content.replace("`", "\\`")

            report += f"## Чанк {i} [Позиция: {position_info}]\n\n"
            report += f"**Метаданные:**\n{metadata_str}\n\n"

            # Информация о перекрытии с предыдущим чанком
            if i > 1:
                overlap = overlaps[i - 2]
                report += f"**Перекрытие с предыдущим чанком:** {overlap} символов\n\n"

            report += f"**Полный текст:**\n```\n{chunk_text}\n```\n\n"

    return report


if __name__ == "__main__":
    from config import config
    import sys

    # Фиксированная директория для обработки
    input_dir = Path(__file__).parent / "test_embeding"
    output_dir = Path(__file__).parent / "test_embeding_result"

    # Проверка существования директории
    if not input_dir.exists():
        print(f"Ошибка: каталог не найден - {input_dir}")
        sys.exit(1)

    def process_files(only_clean: bool, clean_text: bool):
        """Обрабатывает все файлы в директории с заданными параметрами"""
        # Создаем каталог для результатов
        output_dir.mkdir(exist_ok=True)

        # Сбор поддерживаемых файлов
        files = []
        for ext in config.SUPPORTED_EXTENSIONS:
            files.extend(input_dir.glob(f"*{ext}"))

        if not files:
            print(f"В каталоге {input_dir} не найдено поддерживаемых файлов")
            return

        print(f"\nНайдено файлов для обработки: {len(files)}")
        for file_path in files:
            print(f"\n=== Обработка файла: {file_path.name} ===")

            if only_clean:
                # Режим только очистки текста
                manager = EmbeddingManager(config)
                cleaner = LegalTextCleaner()
                raw_text = manager.parse_document(str(file_path))
                cleaned_text = cleaner.clean(raw_text)

                # Сохранение очищенного текста
                output_path = output_dir / f"{file_path.stem}_cleaned.md"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                print(f"Очищенный текст сохранен: {output_path}")
            else:
                # Полный анализ чанкинга
                report = analyze_document(str(file_path), clean_text=clean_text)

                # Сохранение отчета
                suffix = "_cleaned" if clean_text else "_raw"
                filename = f"{file_path.stem}{suffix}_chunk_report.md"
                output_path = output_dir / filename
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"Отчет сохранен: {output_path}")

    # Текстовое меню
    while True:
        print("\n" + "=" * 40)
        print("=== Меню анализатора чанков ===")
        print("=" * 40)
        print("Обрабатываемая директория: test_embeding")
        print("\nВыберите действие:")
        print("1. Только очистка текста")
        print("2. Очистка и чанкинг")
        print("3. Чанкинг без очистки")
        print("4. Выход")

        choice = input("> ").strip()

        if choice == "1":
            process_files(only_clean=True, clean_text=True)
        elif choice == "2":
            process_files(only_clean=False, clean_text=True)
        elif choice == "3":
            process_files(only_clean=False, clean_text=False)
        elif choice == "4":
            print("Выход из программы")
            break
        else:
            print("Некорректный выбор. Введите 1-4.")
