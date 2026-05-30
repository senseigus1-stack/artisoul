<#
.SYNOPSIS
    Скачивает ffmpeg (portable) и распаковывает в папку tools.
.DESCRIPTION
    После выполнения папка ffmpeg будет доступна в текущей сессии.
#>

$ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$toolsDir = ".\tools"
$ffmpegZip = "$toolsDir\ffmpeg.zip"
$ffmpegExtract = "$toolsDir\ffmpeg"

# Создаём папку tools, если её нет
if (-not (Test-Path $toolsDir)) { New-Item -ItemType Directory -Path $toolsDir | Out-Null }

Write-Host "Скачиваю ffmpeg..."
Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegZip

Write-Host "Распаковываю..."
Expand-Archive -Path $ffmpegZip -DestinationPath $toolsDir -Force

# Находим папку bin (обычно ffmpeg-master-latest-win64-gpl/bin)
$binPath = Get-ChildItem -Path $toolsDir -Filter "*/bin" -Directory | Select-Object -First 1
if ($binPath) {
    $env:Path += ";$($binPath.FullName)"
    Write-Host "ffmpeg временно добавлен в PATH: $($binPath.FullName)"
    Write-Host "Проверка версии:"
    & "$($binPath.FullName)\ffmpeg.exe" -version
} else {
    Write-Host "Не удалось найти bin/ffmpeg. Проверьте структуру архива."
}

Remove-Item $ffmpegZip