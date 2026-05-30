<#
.SYNOPSIS
    Полная настройка проекта endocrine_ai на Windows.
.DESCRIPTION
    Устанавливает Python-пакеты, ffmpeg и скачивает модели.
#>

Write-Host "=== Установка Python-зависимостей ==="
pip install -r ..\..\requirements.txt

Write-Host "=== Установка ffmpeg ==="
.\install_ffmpeg.ps1

Write-Host "=== Скачивание моделей ==="
.\download_models.ps1

Write-Host "=== Готово! Теперь можно запускать проект ==="