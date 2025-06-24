# langgraph_nodes.py - Fixed backup model integration
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel
import torch
import numpy as np

MODEL_DIR = "A:/intership/task 2/model"

# Load tokenizer and model
base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=28)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load zero-shot pipeline as backup
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
    "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral"
]


def inference_node(state):
    text = state["input"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = torch.sigmoid(outputs.logits)[0]  # Multi-label

    threshold = 0.3
    predicted_indices = (logits > threshold).nonzero(as_tuple=True)[0].tolist()
    if not predicted_indices:
        predicted_indices = [torch.argmax(logits).item()]

    labels = [emotion_labels[i] for i in predicted_indices]
    state["initial_labels"] = labels
    state["max_conf"] = torch.max(logits).item()
    state["primary_model_used"] = True
    return state


def confidence_check_node(state):
    # Always check confidence, but don't set final_labels yet
    if state["max_conf"] < 0.5:
        state["fallback_needed"] = True
    else:
        state["fallback_needed"] = False
    return state


def backup_model_inference(text):
    """Use zero-shot classifier as backup model"""
    try:
        result = zero_shot(text, emotion_labels)
        # Get top predictions with confidence > 0.15
        backup_labels = []
        backup_scores = []

        for label, score in zip(result['labels'], result['scores']):
            if score > 0.15:  # Lower threshold for backup
                backup_labels.append(label)
                backup_scores.append(score)
                if len(backup_labels) >= 3:  # Limit to top 3
                    break

        if not backup_labels:
            backup_labels = [result['labels'][0]]  # At least one label
            backup_scores = [result['scores'][0]]

        return backup_labels, max(backup_scores)
    except Exception as e:
        print(f"❌ Backup model failed: {e}")
        return ["neutral"], 0.3


def fallback_node(state):
    # If already clarified by user, don't use backup model
    if state.get("clarified", False):
        state["fallback_needed"] = False
        state["final_labels"] = state.get("initial_labels", [])
        return state

    # Try backup model
    text = state.get("original_input", state["input"])
    backup_labels, backup_conf = backup_model_inference(text)

    state["backup_labels"] = backup_labels
    state["backup_conf"] = backup_conf
    state["backup_model_used"] = True

    # Decision logic: Use backup if significantly better
    primary_conf = state["max_conf"]
    confidence_threshold = 0.65  # Higher threshold for backup to take over

    if backup_conf > confidence_threshold and backup_conf > primary_conf + 0.1:
        # Backup model is confident and notably better
        state["final_labels"] = backup_labels
        state["max_conf"] = backup_conf  # Update confidence to backup's
        state["fallback_needed"] = False
        state["fallback_method"] = "backup_model"
        print(f"✨ Backup model provided higher confidence solution!")
    else:
        # Neither model is confident enough, need user clarification
        state["fallback_needed"] = True
        state["final_labels"] = state.get("initial_labels", [])

    return state