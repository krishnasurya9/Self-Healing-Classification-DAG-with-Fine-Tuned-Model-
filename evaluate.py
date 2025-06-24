import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# === Paths ===
MODEL_DIR = "A:/intership/task 2/model"
DATA_PATH = "A:/intership/task 2/data/processed/dev"

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=28,
    problem_type="multi_label_classification"
)
model.eval()

# === Load dataset ===
dataset = load_from_disk(DATA_PATH)

# === Tokenize inputs ===
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(preprocess, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# === Collect true labels ===
true_labels = []
for i in range(28):
    true_labels.append(dataset[f"label_{i}"])
true_labels = np.array(true_labels).T  # shape: (num_samples, 28)

# === Inference ===
predictions = []
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        inputs = {
            "input_ids": item["input_ids"].unsqueeze(0),
            "attention_mask": item["attention_mask"].unsqueeze(0)
        }
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().numpy()
        pred = (probs > 0.5).astype(int)
        predictions.append(pred)

predictions = np.array(predictions)

# === Evaluation ===
f1_micro = f1_score(true_labels, predictions, average="micro")
f1_macro = f1_score(true_labels, predictions, average="macro")

print(f"\n✅ Evaluation Metrics:")
print(f"F1 Score (micro): {f1_micro:.4f}")
print(f"F1 Score (macro): {f1_macro:.4f}")

# === Optional: per-label performance
report = classification_report(true_labels, predictions, zero_division=0, target_names=[
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
])

print("\n📊 Per-Label Report:\n")
print(report)
