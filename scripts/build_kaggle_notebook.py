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

# Second notebook: draft fine-tuning, pure-greedy re-evaluation, GPU benchmarks, final test.
REPO = "ibrahimmlk/trialsieve-qwen2.5-1.5b-lora"
DRAFT_REPO = "ibrahimmlk/trialsieve-qwen2.5-0.5b-draft-lora"
FINAL = [
    ("md", "# TrialSieve: benchmarks and final evaluation\nSettings: **GPU T4 x2**, Internet **On**, secret `HF_TOKEN`. Run sections 0-4 top to bottom; section 5 only when told."),
    ("code", "!rm -rf /kaggle/working/trialsieve && git clone -q https://github.com/ibrahimmlk/trialsieve.git /kaggle/working/trialsieve\n%cd /kaggle/working/trialsieve\n!git log --oneline -1\n!pip install -q -U peft pytest\n!pip uninstall -y -q torchao\nimport os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'\nfrom kaggle_secrets import UserSecretsClient\nfrom huggingface_hub import login, HfApi, snapshot_download\nlogin(UserSecretsClient().get_secret('HF_TOKEN'))\napi = HfApi()\n" + f"REPO, DRAFT_REPO = '{REPO}', '{DRAFT_REPO}'\n" + "snapshot_download(REPO, local_dir='outputs/lora-qwen1.5b/final', allow_patterns=['adapter_*', '*.json', '*.jinja'])"),
    ("md", "## 0. Unit tests (attention, manual model vs HF, KV cache, speculative)"),
    ("code", "!python -m pytest -q tests"),
    ("md", "## 1. Re-evaluate validation with pure greedy decoding (repetition penalty off)"),
    ("code", "!python scripts/evaluate.py --split val --tag base\n!python scripts/evaluate.py --split val --adapter outputs/lora-qwen1.5b/final --tag finetuned"),
    ("md", "## 2. Fine-tune the 0.5B draft model on the same train split (for speculative decoding)"),
    ("code", "!python scripts/train.py --model Qwen/Qwen2.5-0.5B-Instruct --output-dir outputs/lora-qwen0.5b-draft\napi.create_repo(DRAFT_REPO, exist_ok=True)\napi.upload_folder(repo_id=DRAFT_REPO, folder_path='outputs/lora-qwen0.5b-draft/final')\nfor f in ['log_history.json', 'train_config.json']:\n    api.upload_file(repo_id=DRAFT_REPO, path_or_fileobj=f'outputs/lora-qwen0.5b-draft/{f}', path_in_repo=f'training/{f}')"),
    ("md", "## 3. GPU benchmarks: KV cache, attention kernels, speculative decoding"),
    ("code", "!python scripts/benchmark.py --part all --adapter outputs/lora-qwen1.5b/final --draft-adapter outputs/lora-qwen0.5b-draft/final"),
    ("md", "## 4. Upload results"),
    ("code", "api.upload_folder(repo_id=REPO, folder_path='results', path_in_repo='results')\nprint('done')"),
    ("md", "## 5. FINAL held-out test evaluation (run once, only when told)"),
    ("code", "!python scripts/evaluate.py --split test --final --tag base\n!python scripts/evaluate.py --split test --final --adapter outputs/lora-qwen1.5b/final --tag finetuned\napi.upload_folder(repo_id=REPO, folder_path='results', path_in_repo='results')"),
]
nb["cells"] = [{"cell_type": "markdown" if t == "md" else "code", "metadata": {}, "source": src,
                **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, src in FINAL]
Path(__file__).resolve().parents[1].joinpath("notebooks", "kaggle_benchmarks_final.ipynb").write_text(json.dumps(nb, indent=1))
