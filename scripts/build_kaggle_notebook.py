"""Writes notebooks/kaggle_finetune.ipynb (kept as a script so the notebook is reproducible)."""
import json
from pathlib import Path

CELLS = [
    ("md", "# TrialSieve: LoRA fine-tuning on Kaggle\nSettings: Accelerator **GPU T4 x2**, Internet **On**, secret `HF_TOKEN` attached. Run cells top to bottom."),
    ("code", "!rm -rf trialsieve && git clone -q https://github.com/ibrahimmlk/trialsieve.git\n%cd trialsieve\n!git log --oneline -1\n!pip install -q -U peft\n!pip uninstall -y -q torchao  # Kaggle's torchao 0.10 makes peft refuse to load; it is unused here\nimport os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # single T4\n!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    ("code", "from kaggle_secrets import UserSecretsClient\nfrom huggingface_hub import login\nlogin(UserSecretsClient().get_secret('HF_TOKEN'))"),
    ("md", "## 1. Base model on validation (before fine-tuning)"),
    ("code", "!python scripts/evaluate.py --split val --tag base"),
    ("md", "## 2. LoRA fine-tuning (train split only, validation loss for checkpoint selection)"),
    ("code", "!python scripts/train.py --output-dir outputs/lora-qwen1.5b"),
    ("md", "## 3. Fine-tuned model on validation"),
    ("code", "!python scripts/evaluate.py --split val --adapter outputs/lora-qwen1.5b/final --tag finetuned"),
    ("md", "## 4. Upload adapter and validation results to Hugging Face"),
    ("code", "from huggingface_hub import HfApi\napi = HfApi(); repo = 'ibrahimmlk/trialsieve-qwen2.5-1.5b-lora'\napi.create_repo(repo, exist_ok=True)\napi.upload_folder(repo_id=repo, folder_path='outputs/lora-qwen1.5b/final')\nfor f in ['log_history.json', 'train_config.json']:\n    api.upload_file(repo_id=repo, path_or_fileobj=f'outputs/lora-qwen1.5b/{f}', path_in_repo=f'training/{f}')\napi.upload_folder(repo_id=repo, folder_path='results', path_in_repo='results')\nprint('https://huggingface.co/' + repo)"),
    ("md", "## 5. FINAL held-out test evaluation\nRun **only** once the model and prompt are frozen from validation results. Never tune after this."),
    ("code", "!python scripts/evaluate.py --split test --final --tag base\n!python scripts/evaluate.py --split test --final --adapter outputs/lora-qwen1.5b/final --tag finetuned\napi.upload_folder(repo_id=repo, folder_path='results', path_in_repo='results')"),
]

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
      "cells": [{"cell_type": "markdown" if t == "md" else "code", "metadata": {}, "source": src,
                 **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, src in CELLS]}
Path(__file__).resolve().parents[1].joinpath("notebooks", "kaggle_finetune.ipynb").write_text(json.dumps(nb, indent=1))
