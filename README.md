# Self-Healing-Classification-DAG-with-Fine-Tuned-Model-
A multi-label emotion classifier with self-healing logic. Built with DistilBERT and LangGraph, it uses a stateful workflow with fallbacks (backup zero-shot model, user clarification) for low-confidence predictions. Comes with tools for performance analytics and visualization
## Data and Model Setup

The necessary data and the fine-tuned model are provided in `data.zip` and `model.zip` respectively, located in the repository's root.

**To prepare these files:**

1.  **Extract:** Unzip `data.zip` into a folder named `data/` and `model.zip` into a folder named `model/` directly within this project's root directory.
    * *Example (Linux/macOS):* `unzip data.zip -d data/` and `unzip model.zip -d model/`
2.  **Pathing:** All project scripts (`preprocess.py`, `train_lora.py`, etc.) are designed to load these resources from the `data/` and `model/` subdirectories. If you extract them elsewhere, you will need to adjust the file paths in the relevant Python scripts accordingly.
