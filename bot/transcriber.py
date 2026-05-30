"""
Transcription of voice messages using OpenAI Whisper.
"""

import whisper
from .config import WHISPER_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL)
    return _model

def transcribe_audio(file_path: str) -> str:
    """Transcribe the given audio file. Returns recognized text."""
    model = get_model()
    result = model.transcribe(file_path, language="ru")
    return result["text"].strip()