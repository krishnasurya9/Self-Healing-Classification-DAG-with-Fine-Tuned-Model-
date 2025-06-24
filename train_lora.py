import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model

# ✅ Custom Trainer to support multi-label BCE loss
class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float().to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ✅ Load preprocessed datasets
train_dataset = load_from_disk("A:/intership/task 2/data/processed/train")
eval_dataset = load_from_disk("A:/intership/task 2/data/processed/dev")

# ✅ Load tokenizer and base model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=28,
    problem_type="multi_label_classification"
)

# ✅ Apply LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)
model = get_peft_model(base_model, lora_config)

# ✅ Tokenize
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

# ✅ Group labels: label_0 to label_27 → labels list
def group_labels(example):
    example["labels"] = [example[f"label_{i}"] for i in range(28)]
    return example

train_dataset = train_dataset.map(group_labels)
eval_dataset = eval_dataset.map(group_labels)

# ✅ Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# ✅ Metrics
def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = labels.astype(int)
    preds = (probs > 0.5).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# ✅ Training arguments
training_args = TrainingArguments(
    output_dir="A:/intership/task 2/model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    logging_dir="A:/intership/task 2/logs",
    logging_steps=100,
    save_steps=500,
    save_total_limit=1,
)

# ✅ Trainer
trainer = MultiLabelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# ✅ Train
trainer.train()

# ✅ Save model
model.save_pretrained("A:/intership/task 2/model")
tokenizer.save_pretrained("A:/intership/task 2/model")

print("✅ Training complete. Model saved to A:/intership/task 2/model")
