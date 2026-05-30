# fine_tune.py
# Fine-tunes a Russian language model on transcripts obtained from audio.
# Expected input: data/transcripts.json – a list of {"filename": ..., "text": ...}

import json
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

def load_transcripts(json_path: str) -> Dataset:
    """Load transcript texts from a JSON file into a HuggingFace Dataset."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data if item["text"].strip()]
    print(f"Loaded {len(texts)} transcripts")
    return Dataset.from_dict({"text": texts})

def tokenize_function(examples, tokenizer):
    """Tokenize the text field."""
    return tokenizer(examples["text"], truncation=True, max_length=256)

def main():
    # Configuration
    base_model = "ai-forever/rugpt3small"       # or your local model path
    output_dir = "./fine_tuned_sterfa"           # where to save the fine-tuned model
    json_path = "data/transcripts.json"

    if not Path(json_path).exists():
        print(f"File {json_path} not found. Run transcribe.py first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_transcripts(json_path)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
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
    print(f"Fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    main()