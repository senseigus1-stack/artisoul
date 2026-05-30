# transcribe.py
# Transcribes Russian speech from MP3 files using OpenAI Whisper,
# deletes the source files after successful recognition,
# and stores the text in data/transcripts.json.

import os
import json
import whisper
from pathlib import Path
import torch

def transcribe_audio(audio_path: str, model) -> str:
    """Transcribe a single audio file. Returns the recognized text."""
    result = model.transcribe(audio_path, language="ru", fp16=torch.cuda.is_available())
    return result["text"].strip()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Whisper model (choose size: tiny, base, small, medium, large)
    model = whisper.load_model("medium", device=device)

    audio_dir = Path("data/audio")
    if not audio_dir.exists():
        audio_dir.mkdir(parents=True)
        print(f"Directory {audio_dir} created. Put your MP3 files there and re-run.")
        return

    mp3_files = list(audio_dir.glob("*.mp3"))
    if not mp3_files:
        print("No .mp3 files found in data/audio.")
        return

    transcripts = []

    for mp3_path in mp3_files:
        print(f"Processing {mp3_path.name} ...")
        try:
            text = transcribe_audio(str(mp3_path), model)
            transcripts.append({"filename": mp3_path.name, "text": text})
            # Remove the file after successful transcription
            os.remove(mp3_path)
            print(f"  -> transcribed and removed.")
        except Exception as e:
            print(f"  Error: {e}")

    # Save all transcripts
    output_file = Path("data/transcripts.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(transcripts)} files. Output saved to {output_file}")

if __name__ == "__main__":
    main()