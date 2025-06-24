🔮 Emotion Classifier with LangGraph Workflow
A resilient and intelligent multi-label emotion classification system built with a fine-tuned DistilBERT model and LangGraph. This project implements self-healing logic and robust fallback strategies to handle low-confidence predictions, providing a more reliable classification experience.  Integrated visual analytics offer deep insights into model performance and workflow dynamics. 


✨ Key Features
This system boasts a range of powerful features to ensure accurate and reliable emotion classification:

Fine-tuned DistilBERT with LoRA: Leverages a highly optimized DistilBERT model, fine-tuned efficiently using Low-Rank Adaptation (LoRA) for superior performance on emotion classification tasks. 
LangGraph-Powered Workflow: Implements a sophisticated, stateful workflow using LangGraph for dynamic and intelligent decision-making: 
Inference Node: Performs the initial emotion prediction. 
Confidence Check Node: Evaluates prediction confidence to identify ambiguous cases. 
Fallback Node: Activates intelligent fallback strategies when confidence is low. 
Robust Fallback Strategies: Ensures reliable classification even in challenging scenarios: 
User Clarification: Prompts the user for additional context to resolve ambiguity. 
Backup Zero-Shot Model: Routes low-confidence predictions to a secondary, more generalized zero-shot model (facebook/bart-large-mnli) for a second opinion. 
Comprehensive, Structured Logging: Every prediction is meticulously logged in JSON format, capturing timestamps, input, initial and final predictions, confidence scores, and detailed fallback information for thorough analysis. 
Insightful Analytics Dashboard: Provides crucial insights into system behavior and performance: 
Confidence Trends: Visualize model confidence over time. 
Fallback Frequency: Track how often and which fallback methods are triggered. 
CLI + Chart Visualizations: Generate both in-terminal metrics and high-quality plots. 
📂 Project Structure
Your project is meticulously organized for clarity and maintainability:

A:/intership/task 2/
├── data/                  ← Input dataset (e.g., TSV/CSV) 
├── model/                 ← Fine-tuned model checkpoints + tokenizer 
├── logs/                  ← Automated log file for predictions (auto-created) 
├── scripts/               ← Core Python modules for application logic 
│   ├── preprocess.py      ← Dataset cleaning and preparation scripts 
│   ├── train_lora.py      ← Script for fine-tuning the base model using LoRA 
│   ├── langgraph_nodes.py ← Implements inference and sophisticated fallback nodes 
│   ├── graph.py           ← Constructs the LangGraph Directed Acyclic Graph (DAG) 
│   ├── cli.py             ← Command-Line Interface for interactive emotion testing 
│   └── statistics_tracker.py← Tools for visualizing and tracking performance statistics 
└── README.md              ← Comprehensive overview and usage guide (this file) 
⚙️ Setup Instructions
Follow these steps to get the Emotion Classifier up and running:

Clone the Repository:

Bash

git clone https://github.com/your-username/emotion-classifier-langgraph.git
cd emotion-classifier-langgraph
(Note: Replace your-username/emotion-classifier-langgraph.git with your actual repository URL) 

Create a Virtual Environment (Recommended):

Bash

python -m venv .venv
Activate it:

Linux/macOS: source .venv/bin/activate 
Windows: .venv\Scripts\activate 
Install Dependencies:

Bash

pip install -r requirements.txt
Preprocess the Data:
Before training, process the raw .tsv files.

Bash

python scripts/preprocess.py
Train the Model with LoRA:
Fine-tune the DistilBERT model on your dataset:

Bash

python scripts/train_lora.py
This step will download the base model, train it, and save the fine-tuned checkpoints to the model/ directory. 

⚠️ Important: Update File Paths
Before running the scripts, you must update the hardcoded absolute file paths to match your system's directory structure. It is highly recommended to use relative paths to ensure portability.

The following files contain hardcoded paths like A:/intership/task 2/...:

scripts/preprocess.py
scripts/train_lora.py
scripts/langgraph_nodes.py
scripts/cli.py
scripts/statistics_tracker.py
evaluate.py
Example:
Change a hardcoded path:

Python

# In train_lora.py
# Before
train_dataset = load_from_disk("A:/intership/task 2/data/processed/train")
output_dir="A:/intership/task 2/model"
to a relative path:

Python

# After
train_dataset = load_from_disk("data/processed/train")
output_dir="model"
🧠 Using the CLI
Engage with the emotion classifier interactively via the Command-Line Interface: 

Bash

python scripts/cli.py
Upon launch, you'll see a list of commands for classifying text and viewing statistics. 

🔬 Model Evaluation
To evaluate the performance of your fine-tuned model against the test set, run the evaluation script. This will compute F1 scores (micro and macro) and generate a detailed per-label classification report. 

Bash

python evaluate.py
📈 Analytics & Visualizations
Gain deep insights into your model's performance and the LangGraph workflow. Generate CLI statistics and visual graphs by running:

Bash

python scripts/statistics_tracker.py
This script will display metrics in the terminal and save the following plots to your root directory:

📊 confidence_curve.png: A plot showing confidence scores over time and their distribution. 
📊 fallback_stats.png: Visualizations of fallback rates, the methods used, and overall usage statistics. 

🧪 Test Cases
You can manually test the system with various inputs to observe its behavior: 

High confidence emotional input:
I am absolutely thrilled with the results, this is fantastic! 
I am beyond furious with this blatant disregard! 
A wave of sadness washed over me as I heard the news. 
Neutral input:
The sky is blue today. 
This movie was neither good nor bad, just okay. 
Low confidence / Ambiguous input (should trigger fallback):
This situation is quite frustrating and confusing. 
Feeling a bit off today, can't quite put my finger on it. 
📚 Detailed Logging Format
Each prediction event is meticulously logged as a JSON object in logs/langgraph_interactions.log, providing a rich dataset for analysis: 

JSON

{
  "timestamp": "2025-06-24T16:43:00.123456",
  "input": "I'm annoyed by this situation, it's really getting to me.",
  "initial_labels": ["annoyance"],
  "initial_conf": 0.47,
  "fallback_triggered": true,
  "final_labels": ["anger"],
  "fallback_method": "user_clarification",
  "clarified_input": "frustrated and angry because of the situation"
}
🏁 Final Notes
This project stands as a robust example of integrating advanced NLP models with dynamic, stateful orchestration via LangGraph.  It provides a foundation for building intelligent systems that can adapt and self-correct, offering greater reliability than traditional linear pipelines. 


Designed with extensibility in mind, this architecture can be readily expanded to incorporate: 

Multi-modal input (text, voice, video). 
Voice analysis and speech-to-text. 
Advanced user interfaces beyond the CLI. 
🤝 Author
P. Krishna Surya