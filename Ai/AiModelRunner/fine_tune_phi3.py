from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import time
import shutil
import os

# Path to the fine-tuned model
output_dir = "./phi3_finetuned_long"

# Check if the directory exists, and delete it
if os.path.exists(output_dir):
    print(f"🗑️ Deleting existing fine-tuned model: {output_dir}")
    shutil.rmtree(output_dir)

# Load model & tokenizer
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset_path = "habits_train.jsonl"  # Make sure this file exists!
print(f"Loading dataset from: {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path)

# Tokenize dataset
def tokenize_function(example):
    inputs = tokenizer(example["input"], padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(example["output"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Fine-tuning settings
training_args = TrainingArguments(
    output_dir="./phi3_finetuned_long",  # Directory to save model
    per_device_train_batch_size=4,  # Adjust batch size based on available VRAM
    num_train_epochs=20,  # Run for longer training
    save_strategy="epoch",  # Save after each epoch
    save_total_limit=5,  # Keep last 5 checkpoints
    logging_dir="./logs",  # Directory for logs
    logging_steps=50,  # Log training progress
    learning_rate=2e-5,  # Lower learning rate for better fine-tuning
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU supports it
    gradient_accumulation_steps=16,  # Helps smaller GPUs handle training
    optim="adamw_torch",
    report_to="none",  # Disable logging to Hugging Face
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)

# Start training and track time
start_time = time.time()
print("🚀 Starting fine-tuning...")

trainer.train()

# Save the fine-tuned model
trainer.save_model("./phi3_finetuned_long")
tokenizer.save_pretrained("./phi3_finetuned_long")

# Print total training time
end_time = time.time()
elapsed_time = (end_time - start_time) / 60
print(f"✅ Fine-tuning complete! Model saved to './phi3_finetuned_long' (Training took {elapsed_time:.2f} minutes).")
