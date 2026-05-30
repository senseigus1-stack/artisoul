# sentiment_analyzer.py
# Extracts stress (negative) and comfort (positive) signals from user text.
# Uses a pretrained Russian sentiment model with a multilingual fallback.

from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        # Try Russian sentiment model first
        try:
            self.pipe = pipeline(
                "sentiment-analysis",
                model="blanchefort/rubert-base-cased-sentiment",
                return_all_scores=True
            )
            self.language = 'ru'
        except Exception:
            # Fallback to multilingual
            self.pipe = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                return_all_scores=True
            )
            self.language = 'multi'

    def analyze(self, text: str):
        """Return (stress_signal, comfort_signal) both in [0, 1]."""
        if not text.strip():
            return 0.0, 0.0
        try:
            result = self.pipe(text[:512])[0]
        except Exception:
            return 0.0, 0.0

        if self.language == 'ru':
            neg = next((item['score'] for item in result if item['label'] == 'NEGATIVE'), 0)
            pos = next((item['score'] for item in result if item['label'] == 'POSITIVE'), 0)
            return neg, pos
        else:
            # Multilingual model returns star ratings 1..5
            stars = {item['label']: item['score'] for item in result}
            neg = stars.get('1 star', 0) + stars.get('2 stars', 0) * 0.5
            pos = stars.get('4 stars', 0) * 0.5 + stars.get('5 stars', 0)
            return min(1.0, neg), min(1.0, pos)