from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load model & tokenizer
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("json", data_files="habits_train.jsonl")

# Tokenize dataset
def tokenize_function(example):
    inputs = tokenizer(example["input"], padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(example["output"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Longer Training Arguments
training_args = TrainingArguments(
    output_dir="./phi3_finetuned_long",  # Save location
    per_device_train_batch_size=2,  # Adjust batch size based on available VRAM
    num_train_epochs=10,  # Increase epochs for deeper training
    save_strategy="epoch",  # Save checkpoint after each epoch
    save_total_limit=3,  # Keep last 3 checkpoints
    evaluation_strategy="no",  # No evaluation step (optional)
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=3e-5,  # Lower learning rate for better fine-tuning
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU supports it
    gradient_accumulation_steps=8,  # Accumulate gradients for stable training
    optim="adamw_torch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
trainer.save_model("./phi3_finetuned_long")
tokenizer.save_pretrained("./phi3_finetuned_long")

print("✅ Long fine-tuning complete! Model saved to './phi3_finetuned_long'")
