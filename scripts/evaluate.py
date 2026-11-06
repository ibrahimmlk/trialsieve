"""Greedy-decode a split with the base model or base+adapter and compute metrics.

The test split is held out: it is only evaluated when --final is passed.

    python scripts/evaluate.py --split val --tag base
    python scripts/evaluate.py --split val --adapter outputs/lora-qwen1.5b/final --tag finetuned
    python scripts/evaluate.py --split test --final --adapter ... --tag finetuned
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trialsieve.data import gold_evidence, load_split, messages, note_sentences, parse_output  # noqa: E402
from trialsieve.metrics import full_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="required to touch the held-out test split")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--limit", type=int, default=None, help="debug: evaluate only the first N records")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("Refusing to evaluate the held-out test split without --final.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    records = load_split(args.split)[: args.limit]
    prompts = [tok.apply_chat_template(messages(r), tokenize=False, add_generation_prompt=True) for r in records]
    outputs, t0 = [], time.time()
    for i in range(0, len(prompts), args.batch_size):
        batch = tok(prompts[i:i + args.batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**batch, do_sample=False, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
        outputs += tok.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"{min(i + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)
    elapsed = time.time() - t0

    preds = []
    for r, text in zip(records, outputs):
        label, ids, ok = parse_output(text, r["criterion_type"], len(note_sentences(r["note"])))
        preds.append({"annotation_id": r["annotation_id"], "patient_id": r["patient_id"], "trial_id": r["trial_id"],
                      "criterion_type": r["criterion_type"], "criterion_text": r["criterion_text"],
                      "gold_label": r["expert_eligibility"], "label": label, "parsed_json": ok,
                      "gold_evidence": gold_evidence(r), "evidence": ids, "raw_output": text})

    report = full_report(records, preds)
    report["config"] = {"model": args.model, "adapter": args.adapter, "split": args.split, "decoding": "greedy",
                        "max_new_tokens": args.max_new_tokens, "dtype": str(dtype), "device": device,
                        "n": len(records), "seconds": elapsed}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{args.tag}"
    with open(out / f"{stem}_predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / f"{stem}_metrics.json").write_text(json.dumps(report, indent=2))
    a = report["all"]
    print(f"[{stem}] acc={a['classification']['accuracy']:.3f} macroF1={a['classification']['macro_f1']:.3f} "
          f"evidence microF1={a['evidence']['micro']['f1']:.3f} invalid={a['classification']['invalid_outputs']}")


if __name__ == "__main__":
    main()
