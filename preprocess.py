import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset

# Emotion list
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
    "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral"
]
NUM_LABELS = len(emotion_labels)
label_cols = [f"label_{i}" for i in range(NUM_LABELS)]

# File paths
base_path = "A:/intership/task 2/data/archive/data/"
file_map = {
    "train": base_path + "train.tsv",
    "dev": base_path + "dev.tsv",
    "test": base_path + "test.tsv"
}

# Function to load and process each split
def load_and_process_tsv(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["text", "labels", "comment_id"])
    df["labels"] = df["labels"].apply(lambda x: list(map(int, x.split(","))))
    mlb = MultiLabelBinarizer(classes=list(range(NUM_LABELS)))
    multi_hot = mlb.fit_transform(df["labels"])
    label_df = pd.DataFrame(multi_hot, columns=label_cols)
    df = pd.concat([df[["text"]], label_df], axis=1)
    return Dataset.from_pandas(df)

# ✅ Load datasets
train_dataset = load_and_process_tsv(file_map["train"])
dev_dataset   = load_and_process_tsv(file_map["dev"])
test_dataset  = load_and_process_tsv(file_map["test"])

# ✅ Save datasets
train_dataset.save_to_disk("A:/intership/task 2/data/processed/train")
dev_dataset.save_to_disk("A:/intership/task 2/data/processed/dev")
test_dataset.save_to_disk("A:/intership/task 2/data/processed/test")

