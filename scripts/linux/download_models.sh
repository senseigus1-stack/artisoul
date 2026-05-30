#!/bin/bash
# Загрузка русскоязычных LLM из Hugging Face
set -e

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

MODELS=(
    "ai-forever/rugpt3large"
if ! command -v git-lfs &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y git-lfs
    git lfs install
fi

for MODEL in "${MODELS[@]}"; do
    echo "Скачиваю $MODEL ..."
    huggingface-cli download "$MODEL" --local-dir "$MODELS_DIR/$MODEL" --resume-download
done

echo "Готово. Модели сохранены в $MODELS_DIR"