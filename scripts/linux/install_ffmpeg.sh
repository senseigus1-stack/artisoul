#!/bin/bash
# Установка ffmpeg для Ubuntu/Debian
set -e

echo "Установка ffmpeg..."
sudo apt-get update
sudo apt-get install -y ffmpeg

echo "Проверка версии:"
ffmpeg -version