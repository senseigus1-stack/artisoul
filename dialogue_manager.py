# dialogue_manager.py
# Stores dialogue history and optionally compresses older parts via summarization
# to stay within the model's token limit.

from transformers import pipeline, AutoTokenizer

class DialogueManager:
    def __init__(self, max_history_tokens=300, summarize_old=True,
                 summarization_model="facebook/bart-large-cnn", tokenizer=None):
        self.max_tokens = max_history_tokens
        self.summarize_old = summarize_old
        self.turns = []                # list of dicts: {'speaker': ..., 'text': ...}
        self.tokenizer = tokenizer

        if summarize_old:
            try:
                self.summarizer = pipeline("summarization", model=summarization_model)
            except Exception:
                self.summarizer = None
        else:
            self.summarizer = None

    def add_turn(self, speaker, text):
        """Add a new utterance and compress history if necessary."""
        self.turns.append({'speaker': speaker, 'text': text})
        self._compress_if_needed()

    def _compress_if_needed(self):
        if not self.turns:
            return
        full_text = "\n".join(f"{t['speaker']}: {t['text']}" for t in self.turns)
        # Estimate token count
        if self.tokenizer:
            tokens = len(self.tokenizer.encode(full_text))
        else:
            tokens = len(full_text) // 4   # rough approximation

        if tokens > self.max_tokens and len(self.turns) > 4:
            # Compress the older 70% of turns
            split_idx = max(2, int(len(self.turns) * 0.7))
            old_turns = self.turns[:split_idx]
            new_turns = self.turns[split_idx:]
            old_text = "\n".join(f"{t['speaker']}: {t['text']}" for t in old_turns)

            if self.summarizer:
                try:
                    summary = self.summarizer(old_text, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
                except Exception:
                    summary = old_text[-200:]
            else:
                summary = old_text[-200:]

            self.turns = [{'speaker': 'Summary', 'text': summary}] + new_turns

    def get_context(self):
        """Return the entire dialogue history as a single string."""
        return "\n".join(f"{t['speaker']}: {t['text']}" for t in self.turns)