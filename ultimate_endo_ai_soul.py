"""
Ultimate Endocrine Conversational AI

This script combines a biologically detailed endocrine system simulation
with a transformer language model fine‑tuned on conversational data.
The AI responds like a normal human, with emotional states driven by hormone dynamics.
Uses the `datasets` library; training data now supplied as JSON.
"""

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import load_dataset
import time
import random
from math import exp, sin, pi

# ============================================================
# ULTRA-DETAILED ENDOCRINE SYSTEM MODEL
# ============================================================
class UltimateEndocrineSystem:
    """
    Simulates multiple hormonal axes and their interactions:
    - HPA axis (CRH → ACTH → cortisol) with glucocorticoid receptor adaptation
    - Serotonergic system with 5-HT1A autoreceptors
    - Dopaminergic system (tonic and phasic)
    - Noradrenaline (fast stress response)
    - Oxytocin (social bonding)
    - Thyroid axis (T3 – metabolic rate)
    - HPG axis (GnRH → LH → testosterone/estradiol/progesterone)
    - Allostatic load (cumulative wear and tear)
    - Circadian rhythms
    - Glucose/insulin metabolic loop
    """
    def __init__(self, sex='male', baseline_anxiety=0.3, baseline_social=0.5):
        self.sex = sex
        # Constitutional (genetic) factors
        self.baseline_anxiety = baseline_anxiety
        self.baseline_social = baseline_social
        self.baseline_energy = 0.6

        # --- Hormone and neurotransmitter levels (0..1) ---
        self.cortisol = 0.3
        self.serotonin = 0.6
        self.dopamine = 0.5
        self.noradrenaline = 0.2
        self.oxytocin = 0.3
        self.t3 = 0.5                     # triiodothyronine

        # HPG axis components
        self.gnrh = 0.4                   # gonadotropin‑releasing hormone
        self.lh = 0.4                     # luteinizing hormone
        self.testosterone = 0.6 if sex == 'male' else 0.1
        self.estradiol = 0.1 if sex == 'male' else 0.5
        self.progesterone = 0.2 if sex == 'female' else 0.0

        # Intermediate HPA axis nodes
        self.crh = 0.3                    # corticotropin‑releasing hormone
        self.acth = 0.3                   # adrenocorticotropic hormone

        # Metabolism
        self.glucose = 0.6
        self.insulin = 0.5

        # --- Adaptive receptor systems ---
        self.gr_sensitivity = 0.8         # glucocorticoid receptor sensitivity (0..1)
        self.ht1a_sensitivity = 0.5       # 5‑HT1A autoreceptor sensitivity

        # --- Allostatic load (0..1) ---
        self.allostatic_load = 0.2

        # --- Time and rhythms ---
        self.last_update = time.time()
        self.circadian_phase = 0.0        # 0..1, 0 = midnight

    def update(self, stress_signal: float, comfort_signal: float, dt: float = None):
        """
        Update hormone levels based on external emotional stimuli.

        Args:
            stress_signal: 0..1, amount of negative/threatening content
            comfort_signal: 0..1, amount of positive/comforting content
            dt: time elapsed since last update in seconds (None uses system clock)
        """
        if dt is None:
            now = time.time()
            dt = now - self.last_update
            self.last_update = now
        dt = min(dt, 600)                 # cap at 10 minutes to avoid huge jumps
        dt_min = dt / 60.0                # work in minutes for convenient scaling

        # === Advanced circadian rhythms ===
        self.circadian_phase = (self.circadian_phase + dt / 86400) % 1.0
        ph = self.circadian_phase
        # Cortisol: main peak ~8 am (ph=0.33), smaller afternoon rise ~4 pm (ph=0.67)
        cort_rhythm = 0.3 + 0.25 * sin(2*pi*(ph - 0.33)) + 0.1 * sin(4*pi*(ph - 0.33))
        # Serotonin: higher during daylight (ph=0.5)
        sero_rhythm = 0.5 + 0.2 * sin(2*pi*(ph - 0.5))
        # Thyroid hormone: dips at night
        t3_rhythm = 0.5 + 0.1 * sin(2*pi*(ph - 0.25))
        # Testosterone: morning peak
        t_rhythm = 0.05 * sin(2*pi*(ph - 0.3))

        # === HPA axis (stress) ===
        allostasis_factor = self.allostatic_load
        # CRH: stimulated by stress signal and allostasis, inhibited by cortisol via GR
        crh_input = (stress_signal * 0.3 + allostasis_factor * 0.1
                     - self.cortisol * self.gr_sensitivity * 0.15)
        self.crh += (crh_input - 0.02 * self.crh) * dt_min
        self.crh = max(0.0, min(1.0, self.crh))

        # ACTH
        acth_input = 0.6 * self.crh - self.cortisol * self.gr_sensitivity * 0.1
        self.acth += (acth_input - 0.03 * self.acth) * dt_min
        self.acth = max(0.0, min(1.0, self.acth))

        # Cortisol: production driven by ACTH and circadian rhythm
        cortisol_prod = 0.4 * self.acth + 0.15 * cort_rhythm + 0.05 * allostasis_factor
        self.cortisol += (cortisol_prod - 0.015 * self.cortisol) * dt_min
        self.cortisol = max(0.0, min(1.0, self.cortisol))

        # === Noradrenaline (fast stress response) ===
        na_change = stress_signal * 0.4 - 0.05 * self.noradrenaline
        self.noradrenaline += na_change * dt_min
        self.noradrenaline = max(0.0, min(1.0, self.noradrenaline))

        # === Serotonergic system ===
        # Synthesis depends on comfort, inhibited by stress; autoreceptors reduce release
        sero_synthesis = (0.1 * comfort_signal + 0.05 * sero_rhythm
                          - 0.05 * stress_signal * (1 + allostasis_factor))
        # 5‑HT1A autoreceptors: activated by serotonin, decrease further release
        self.ht1a_sensitivity += (0.02 * self.serotonin - 0.01 * self.ht1a_sensitivity) * dt_min
        self.ht1a_sensitivity = max(0.0, min(1.0, self.ht1a_sensitivity))
        sero_release = sero_synthesis * (1 - 0.5 * self.ht1a_sensitivity)
        self.serotonin += (sero_release - 0.01 * self.serotonin) * dt_min
        self.serotonin = max(0.0, min(1.0, self.serotonin))

        # === Dopaminergic system ===
        novelty = abs(stress_signal - comfort_signal)
        reward = comfort_signal * 0.8 + 0.2 * novelty   # comfort + unexpected positive
        dopamine_change = 0.1 * reward + 0.05 * novelty - 0.05 * self.cortisol
        self.dopamine += (dopamine_change - 0.015 * self.dopamine) * dt_min
        self.dopamine = max(0.0, min(1.0, self.dopamine))

        # === Oxytocin ===
        oxy_input = comfort_signal * 0.25 * self.baseline_social
        self.oxytocin += (oxy_input - 0.02 * self.oxytocin
                          - 0.1 * self.cortisol * self.oxytocin) * dt_min
        self.oxytocin = max(0.0, min(1.0, self.oxytocin))

        # === Thyroid axis (T3) ===
        t3_input = 0.01 * comfort_signal - 0.02 * stress_signal + 0.02 * t3_rhythm
        self.t3 += (t3_input + 0.01 * (self.baseline_energy - self.t3)) * dt_min
        self.t3 = max(0.0, min(1.0, self.t3))

        # === HPG (gonadal) axis ===
        # GnRH pulsatile (simplified), suppressed by cortisol and allostasis
        gnrh_input = 0.2 * (1 - self.cortisol) - 0.05 * allostasis_factor
        self.gnrh += (gnrh_input - 0.02 * self.gnrh) * dt_min
        self.gnrh = max(0.0, min(1.0, self.gnrh))

        # LH stimulated by GnRH; in females a menstrual‑cycle‑like surge is added
        lh_input = 0.4 * self.gnrh
        if self.sex == 'female':
            menstrual_phase = (ph * 24 / 28) % 1.0   # 28‑day cycle
            lh_surge = 0.5 * exp(-50 * (menstrual_phase - 0.5)**2)
            lh_input += lh_surge
        self.lh += (lh_input - 0.03 * self.lh) * dt_min
        self.lh = max(0.0, min(1.0, self.lh))

        if self.sex == 'male':
            t_change = 0.1 * self.lh + 0.02 * t_rhythm - 0.05 * self.cortisol
            self.testosterone += (t_change - 0.01 * self.testosterone) * dt_min
            self.testosterone = max(0.0, min(1.0, self.testosterone))
        else:
            e_change = 0.1 * self.lh - 0.03 * self.cortisol
            self.estradiol += (e_change - 0.01 * self.estradiol) * dt_min
            prog_input = 0.1 * self.lh * (1 if menstrual_phase > 0.5 else 0.2)
            self.progesterone += (prog_input - 0.02 * self.progesterone) * dt_min
            self.estradiol = max(0.0, min(1.0, self.estradiol))
            self.progesterone = max(0.0, min(1.0, self.progesterone))

        # === Metabolism (glucose/insulin) ===
        self.glucose += (0.1 * self.noradrenaline + 0.02 * stress_signal
                         - 0.02 * self.glucose) * dt_min
        self.insulin += (0.1 * comfort_signal - 0.03 * self.insulin) * dt_min
        self.glucose = max(0.0, min(1.0, self.glucose))
        self.insulin = max(0.0, min(1.0, self.insulin))

        # === Allostatic load ===
        if self.cortisol > 0.6:
            allo_inc = 0.02 * (self.cortisol - 0.5)
        else:
            allo_inc = -0.01
        self.allostatic_load += allo_inc * dt_min
        self.allostatic_load = max(0.0, min(1.0, self.allostatic_load))

        # === Receptor dynamics ===
        # GR downregulation under high cortisol
        if self.cortisol > 0.5:
            self.gr_sensitivity -= 0.01 * dt_min
        else:
            self.gr_sensitivity += 0.005 * dt_min
        self.gr_sensitivity = max(0.2, min(1.0, self.gr_sensitivity))

    def state_description(self) -> str:
        """Return a narrative description of the current internal state."""
        c, s, d = self.cortisol, self.serotonin, self.dopamine
        na, oxy, t3 = self.noradrenaline, self.oxytocin, self.t3
        t = self.testosterone if self.sex == 'male' else self.estradiol
        allo = self.allostatic_load

        anxiety = (c * 0.7 + na * 0.3) * 100
        depression = ((1 - s) * 0.6 + (1 - d) * 0.4) * 100
        irritability = max(0, (c - s) * 100 + na * 30)
        energy = (t3 * 0.5 + self.glucose * 0.3 + d * 0.2) * 100
        social_interest = oxy * 100
        burnout = allo * 100
        libido = t * 100

        if allo > 0.6:
            chronic_state = "chronic exhaustion, apathy, cynicism"
        elif allo > 0.3:
            chronic_state = "accumulated fatigue, reduced motivation"
        else:
            chronic_state = "relative freshness"

        if anxiety > 70:
            anxiety_state = "panicky alertness"
        elif anxiety > 40:
            anxiety_state = "elevated anxiety"
        else:
            anxiety_state = "calm"

        if depression > 70:
            mood_state = "deep sadness"
        elif depression > 40:
            mood_state = "low mood"
        else:
            mood_state = "stable mood"

        desc = (f"[internal state: {anxiety_state} ({anxiety:.0f}%), {mood_state}, "
                f"irritability {irritability:.0f}%, energy {energy:.0f}%, "
                f"social interest {social_interest:.0f}%, burnout {burnout:.0f}%, "
                f"libido {libido:.0f}%, {chronic_state}]")
        return desc

# ============================================================
# SENTIMENT ANALYZER (Transformer‑based)
# ============================================================
class SentimentAnalyzer:
    """
    Uses a pretrained Russian sentiment model (or multilingual fallback)
    to extract stress and comfort signals from the user's message.
    """
    def __init__(self):
        try:
            self.pipe = pipeline(
                "sentiment-analysis",
                model="blanchefort/rubert-base-cased-sentiment",
                tokenizer="blanchefort/rubert-base-cased-sentiment",
                return_all_scores=True
            )
            self.language = 'ru'
        except Exception:
            self.pipe = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                return_all_scores=True
            )
            self.language = 'multi'

    def analyze(self, text: str):
        """Return (stress_signal, comfort_signal) each in [0,1]."""
        if not text.strip():
            return 0.0, 0.0
        result = self.pipe(text[:512])[0]
        if self.language == 'ru':
            neg = next(item['score'] for item in result if item['label'] == 'NEGATIVE')
            pos = next(item['score'] for item in result if item['label'] == 'POSITIVE')
            return neg, pos
        else:
            stars = {item['label']: item['score'] for item in result}
            neg = stars.get('1 star', 0) + stars.get('2 stars', 0) * 0.5
            pos = stars.get('4 stars', 0) * 0.5 + stars.get('5 stars', 0)
            return min(1.0, neg), min(1.0, pos)

# ============================================================
# FINE‑TUNING USING JSON DATA (modern, replaces TextDataset)
# ============================================================
def fine_tune_model_from_json(base_model_name: str, json_path: str, output_dir: str):
    """
    Fine‑tune a causal language model on a JSON file containing conversations.
    Expected JSON format: a list of objects with 'user' and 'bot' keys.
    Example:
        [
          {"user": "Hello!", "bot": "Hi! How are you?"},
          {"user": "I'm fine, thanks.", "bot": "Glad to hear!"}
        ]
    The function converts each turn into a training string:
        "User: {user}\nBot: {bot}"
    and tokenizes it.

    Args:
        base_model_name: Hugging Face model identifier (e.g., 'distilgpt2')
        json_path: Path to the JSON file
        output_dir: Directory to save the fine‑tuned model
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load JSON dataset ----
    dataset = load_dataset('json', data_files=json_path)['train']
    print(f"Loaded {len(dataset)} dialogue turns from {json_path}")

    # ---- Convert each JSON object to a single dialogue string ----
    def format_dialogue(example):
        # Concatenate user and bot utterances into the same format used during inference
        example['text'] = f"User: {example['user']}\nBot: {example['bot']}"
        return example

    dataset = dataset.map(format_dialogue)

    # ---- Tokenize the formatted text ----
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['user', 'bot', 'text'])
    # For a causal LM, labels = input_ids (handled by the data collator)

    # ---- Data collator ----
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False   # causal LM
    )

    # ---- Load model and train ----
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine‑tuned model saved to {output_dir}")

# ============================================================
# CONVERSATIONAL BOT WITH ENDOCRINE SYSTEM
# ============================================================
class EndocrineConversationalBot:
    """
    A human‑like conversational AI whose emotional tone is continuously
    modulated by the detailed endocrine system simulation.
    """
    def __init__(self, model_path: str, sex='male'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading language model from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.endocrine = UltimateEndocrineSystem(sex=sex)
        self.sentiment = SentimentAnalyzer()
        self.dialogue_history = []

    def build_prompt(self, user_input: str, state_str: str) -> str:
        """
        Construct a prompt that tells the model to act as a normal human
        whose current mood is described by the endocrine state.
        """
        system = (
            "You are a friendly, empathetic human having a natural conversation. "
            f"Your current emotional and physical state is: {state_str}. "
            "Reflect this state subtly in your tone and word choice, "
            "but always respond coherently and appropriately."
        )
        history = ""
        for entry in self.dialogue_history[-6:]:
            history += f"{entry['speaker']}: {entry['text']}\n"
        prompt = (
            f"{system}\n\n"
            f"{history}"
            f"User: {user_input}\n"
            f"You:"
        )
        return prompt

    def generate_reply(self, user_input: str) -> tuple:
        """
        Process the user's message, update hormones, and generate a reply.

        Returns:
            (reply_text, state_description)
        """
        stress, comfort = self.sentiment.analyze(user_input)
        self.endocrine.update(stress, comfort)
        state_str = self.endocrine.state_description()

        prompt = self.build_prompt(user_input, state_str)
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Dynamic generation parameters based on dopamine/energy
        temp = 0.7 + self.endocrine.dopamine * 0.3 + self.endocrine.noradrenaline * 0.05
        temp += random.uniform(-0.03, 0.03)
        top_p = 0.85 + self.endocrine.serotonin * 0.1

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=temp,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the new reply
        reply = full_text.split("You:")[-1].split("User:")[0].strip()
        for punct in ['.', '!', '?']:
            last = reply.rfind(punct)
            if last > 0:
                reply = reply[:last+1]
                break

        # Update conversation history
        self.dialogue_history.append({'speaker': 'User', 'text': user_input})
        self.dialogue_history.append({'speaker': 'You', 'text': reply})

        # Micro‑update after speaking (internal dynamics)
        self.endocrine.update(0.0, 0.0, dt=1.0)

        return reply, state_str

# ============================================================
# INTERACTIVE CHAT LOOP
# ============================================================
def main():
    print("=" * 60)
    print(" Endocrine‑modulated conversational AI (normal human persona) ")
    print(" Type 'exit' to quit.")
    print("=" * 60)

    # Path to the (fine‑tuned) model. Use a pretrained model for quick testing,
    # or your own fine‑tuned directory after training.
    model_path = 'distilgpt2'   # change to 'my_finetuned_model' after training

    # ---- Optional: fine‑tune on your own conversational JSON ----
    # 1. Prepare a JSON file 'dialogues.json' with an array of objects:
    #    [ {"user": "Hello!", "bot": "Hi! How are you?"}, ... ]
    # 2. Uncomment the next line and run the script once:
    # fine_tune_model_from_json('distilgpt2', 'dialogues.json', 'my_finetuned_model')

    bot = EndocrineConversationalBot(model_path, sex='male')

    print("\n[Bot wakes up... Current state:]")
    print(bot.endocrine.state_description())

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        reply, state = bot.generate_reply(user_input)
        print(f"\n[Internal state: {state}]")
        print(f"Bot: {reply}")

if __name__ == "__main__":
    main()