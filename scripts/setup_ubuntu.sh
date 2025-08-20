#!/bin/bash

# Обновление пакетного менеджера
sudo apt-get update

# Установка системных зависимостей
sudo apt-get install -y python3-venv poppler-utils tesseract-ocr tesseract-ocr-rus libmagic1

# Создание виртуального окружения
python3 -m venv venv

# Активация виртуального окружения
source venv/bin/activate

# Обновление pip
pip install --upgrade pip

# Установка Python-зависимостей
pip install -r requirements.txt
# pip install unstructured[local-inference]  # Для поддержки PDF/DOCX

echo "Установка завершена! Активируйте виртуальное окружение командой: source venv/bin/activate"
