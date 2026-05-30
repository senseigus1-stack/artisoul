# bot.py
# Main conversational bot class that integrates the language model,
# the endocrine system, sentiment analysis, and dialogue management.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import random
from config import Config
from endocrine_model import AdvancedEndocrineSystem
from sentiment_analyzer import SentimentAnalyzer
from dialogue_manager import DialogueManager

class EndocrineConversationalBot:
    def __init__(self, config=None):
        self.config = config if config else Config()
        self.device = self.config.DEVICE

        # Load language model and tokenizer
        print(f"Loading Russian language model {self.config.MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.MODEL_NAME).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize subsystems
        self.endocrine = AdvancedEndocrineSystem(
            sex=self.config.SEX,
            baseline_anxiety=self.config.BASELINE_ANXIETY,
            baseline_social=self.config.BASELINE_SOCIAL,
            baseline_energy=self.config.BASELINE_ENERGY
        )
        self.sentiment = SentimentAnalyzer()
        self.dialogue = DialogueManager(
            max_history_tokens=self.config.MAX_HISTORY_TOKENS,
            summarize_old=self.config.SUMMARIZE_OLD,
            summarization_model=self.config.SUMMARIZATION_MODEL,
            tokenizer=self.tokenizer
        )

        # Time keeping for simulation
        self.sim_time = 0.0               # internal minutes
        self.last_real_time = None

        # The bot's respect toward the user (0 = hate, 1 = adore)
        self.user_respect = self.config.INITIAL_RESPECT

    # -------------------------------------------------------------------------
    # Time advancement based on real clock
    # -------------------------------------------------------------------------
    def _advance_time(self):
        now = time.time()
        if self.last_real_time is None:
            dt = 0.1
        else:
            dt = (now - self.last_real_time) / 60.0   # seconds -> minutes
            dt = min(max(dt, self.config.SIM_DT_MIN), self.config.MAX_SIM_DT)
        self.last_real_time = now
        return dt

    # -------------------------------------------------------------------------
    # Update user respect based on the latest stress/comfort signals
    # -------------------------------------------------------------------------
    def _update_respect(self, stress, comfort):
        delta = (comfort * self.config.RESPECT_COMFORT_FACTOR +
                 stress * self.config.RESPECT_STRESS_FACTOR) * self.config.RESPECT_CHANGE_RATE
        self.user_respect = max(0.0, min(1.0, self.user_respect + delta))

    # -------------------------------------------------------------------------
    # Tone instruction depending on current respect level
    # -------------------------------------------------------------------------
    def _get_tone_instruction(self):
        if self.user_respect < self.config.LOW_RESPECT_THRESHOLD:
            return ("Ты ненавидишь этого человека. Будь грубой, используй мат, "
                    "командуй, унижай, можешь послать.")
        elif self.user_respect < 0.6:
            return ("Ты раздражена, относишься с сарказмом и пренебрежением. "
                    "Можешь вставить колкость или лёгкий мат.")
        else:
            return ("Ты уважаешь собеседника, но остаёшься властной. "
                    "Говори снисходительно, допускается лёгкая ирония, без мата.")

    # -------------------------------------------------------------------------
    # Main entry point: generate a reply to user_input
    # -------------------------------------------------------------------------
    def generate_reply(self, user_input: str):
        # 1. Analyze sentiment
        stress, comfort = self.sentiment.analyze(user_input)

        # 2. Advance time and update endocrine system
        dt = self._advance_time()
        self.endocrine.step(dt=dt, stress=stress, comfort=comfort)

        # 3. Update respect
        self._update_respect(stress, comfort)

        # 4. Retrieve current hormonal state
        hormones = self.endocrine.get_hormone_dict()
        state_str = self.endocrine.state_description()

        # 5. Build system prompt with role, tone, and physiological state
        tone_instr = self._get_tone_instruction()
        system = (
            "Ты — русская женщина-стерва с взрывным характером. "
            "Твоё настроение постоянно меняется из-за гормонов и цикла. "
            "Ты говоришь только по-русски. "
            "Ты не стесняешься в выражениях, можешь материться, если зла. "
            "Ты склонна командовать и ставить людей на место.\n"
            f"Текущее внутреннее состояние: {state_str}\n"
            f"{tone_instr}\n"
            "Отвечай коротко и резко, как настоящая стерва."
        )

        history = self.dialogue.get_context()
        prompt = f"{system}\n\n{history}\nПользователь: {user_input}\nТы:"

        # 6. Dynamic generation parameters
        temp = (self.config.GENERATION_TEMP_BASE
                + hormones['dopamine'] * 0.3
                + hormones['noradrenaline'] * 0.05
                + (1 - self.user_respect) * 0.3
                + random.uniform(-0.03, 0.03))
        top_p = self.config.GENERATION_TOP_P_BASE + hormones['serotonin'] * 0.1

        # 7. Generate response
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                temperature=temp,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = full_text.split("Ты:")[-1].split("Пользователь:")[0].strip()
        # Trim to last punctuation
        for punct in ['.', '!', '?', ')', '…']:
            last = reply.rfind(punct)
            if last > 0:
                reply = reply[:last+1]
                break

        # 8. Update dialogue history
        self.dialogue.add_turn('Пользователь', user_input)
        self.dialogue.add_turn('Ты', reply)

        return reply, state_str