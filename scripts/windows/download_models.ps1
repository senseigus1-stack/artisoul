<#
.SYNOPSIS
    Скачивает русские LLM с Hugging Face.
#>
param(
    [string]$ModelsDir = ".\models"
)

$models = @(
    "ai-forever/rugpt3large",
)
if (-not (Test-Path $ModelsDir)) {
    New-Item -ItemType Directory -Path $ModelsDir | Out-Null
}

foreach ($model in $models) {
    Write-Host "Скачиваю $model ..."
    huggingface-cli download $model --local-dir "$ModelsDir\$model" --resume-download
}

Write-Host "Модели сохранены в $ModelsDir"