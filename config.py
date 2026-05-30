# config.py
# Central configuration for the Endocrine Conversational AI.
# All tunable parameters are collected here for easy adjustment.

class Config:
    # ---------- Language Model ----------
    MODEL_NAME = "ai-forever/rugpt3large"   # Russian GPT model; change to your fine-tuned path if needed
    DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

    # ---------- Endocrine System Persona ----------
    SEX = "female"                 # "male" or "female"; affects gonadal axis and cycle
    BASELINE_ANXIETY = 0.7         # high baseline anxiety (makes her more irritable)
    BASELINE_SOCIAL = 0.2          # low empathy, more selfish
    BASELINE_ENERGY = 0.6

    # ---------- Hormone-Text Integration ----------
    USE_EMOTION_TOKENS = False     # set True if you have special tokens like <anxious> in your tokenizer
    GENERATION_TEMP_BASE = 0.9     # more chaotic responses
    GENERATION_TOP_P_BASE = 0.9
    MAX_NEW_TOKENS = 120

    # ---------- Dialogue Management ----------
    MAX_HISTORY_TOKENS = 300       # maximum tokens kept in dialogue history (compressed if exceeding)
    SUMMARIZE_OLD = True           # use summarization to compress older parts
    SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

    # ---------- Time Simulation (in minutes) ----------
    SIM_DT_MIN = 0.1               # minimum simulation step
    MAX_SIM_DT = 120.0             # max leap when idle (2 hours)

    # ---------- User Respect Dynamics ----------
    INITIAL_RESPECT = 0.5          # starting respect (0 = hate, 1 = adore)
    RESPECT_CHANGE_RATE = 0.1      # how much respect can change per message
    RESPECT_COMFORT_FACTOR = 1.0   # positive influence of comfort
    RESPECT_STRESS_FACTOR = -1.5   # negative influence of stress (she's more sensitive to rudeness)
    LOW_RESPECT_THRESHOLD = 0.3    # below this, full rudeness and swearing kick in