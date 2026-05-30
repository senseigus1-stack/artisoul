"""
Utility functions for saving/loading progress and dialogues.
"""

import json
import os
from .config import DIALOGUES_FILE, PROGRESS_FILE

def load_json(file_path, default):
    """Load a JSON file, returning a default value if the file doesn't exist."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default

def save_json(file_path, data):
    """Save data as JSON to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_progress():
    """Load progress dict {user_id: last_question_index}."""
    return load_json(PROGRESS_FILE, {})

def save_progress(progress):
    """Save progress dict to file."""
    save_json(PROGRESS_FILE, progress)

def load_dialogues():
    """Load all dialogues."""
    return load_json(DIALOGUES_FILE, [])

def append_dialogue(user_text, bot_question):
    """Append a new dialogue turn."""
    dialogues = load_dialogues()
    dialogues.append({
        "user": user_text,
        "bot": bot_question
    })
    save_json(DIALOGUES_FILE, dialogues)