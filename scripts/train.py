"""LoRA supervised fine-tuning on the train split (loss on the JSON answer only).

Example (Kaggle T4):
    python scripts/train.py --output-dir outputs/lora-qwen1.5b
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trialsieve.data import load_split, messages, target_text  # noqa: E402


def encode(tok, record, max_len):
    prompt = tok.apply_chat_template(messages(record), tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tok(target_text(record) + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + answer_ids
    truncated = len(ids) > max_len
    if truncated:
        # Never cut the answer; trim the prompt from the left (logged and reported).
        ids = ids[-max_len:]
        prompt_len = max_len - len(answer_ids)
    else:
        prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + ids[prompt_len:]
    return {"input_ids": ids, "labels": labels, "length": len(prompt_ids) + len(answer_ids), "truncated": truncated}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id)
        labels = torch.full((len(batch), n), -100)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--output-dir", default="outputs/lora-qwen1.5b")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    train = [encode(tok, r, args.max_len) for r in load_split("train")]
    val = [encode(tok, r, args.max_len) for r in load_split("val")]
    stats = {
        split: {"n": len(rows), "max_tokens": max(r["length"] for r in rows),
                "mean_tokens": sum(r["length"] for r in rows) / len(rows),
                "truncated": sum(r["truncated"] for r in rows)}
        for split, rows in [("train", train), ("val", val)]
    }
    print("Token length stats:", json.dumps(stats))

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine", warmup_ratio=0.05,
        weight_decay=0.0, fp16=torch.cuda.is_available(), logging_steps=5, eval_strategy="steps", eval_steps=10,
        save_strategy="steps", save_steps=10, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, report_to="none", seed=args.seed,
        remove_unused_columns=False, group_by_length=False)
    strip = lambda rows: [{"input_ids": r["input_ids"], "labels": r["labels"]} for r in rows]
    trainer = Trainer(model=model, args=targs, train_dataset=strip(train), eval_dataset=strip(val),
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    model.save_pretrained(out / "final")
    tok.save_pretrained(out / "final")
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    (out / "train_config.json").write_text(json.dumps(vars(args) | {
        "token_stats": stats, "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric, "trainable_params": model.get_nb_trainable_parameters()[0],
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}, indent=2))
    print("Saved adapter to", out / "final")


if __name__ == "__main__":
    main()
