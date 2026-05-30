#!/bin/bash
# Полная настройка проекта endocrine_ai на Linux
set -e

echo "=== Установка системных зависимостей ==="
sudo apt-get update
sudo apt-get install -y python3 python3-pip ffmpeg git-lfs

echo "=== Установка Python-зависимостей ==="
pip install -r ../../requirements.txt

echo "=== Скачивание моделей ==="
bash download_models.sh

echo "=== Готово! Теперь можно запускать проект ==="