"""
Configuration for the Telegram bot.
"""

import os

# Your bot token from @BotFather
BOT_TOKEN = "8055791617:AAERCXoeBHKxB0D1GjMD0O8eTVZVccIR3mQ"

# Whisper model size: "tiny", "base", "small", "medium", "large"
WHISPER_MODEL = "large"

# Data files (paths relative to this config file or absolute)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIALOGUES_FILE = os.path.join(BASE_DIR, "dialogues.json")
PROGRESS_FILE = os.path.join(BASE_DIR, "progress.json")

# Maximum number of questions (derived from questions.py list)
TOTAL_QUESTIONS = 250