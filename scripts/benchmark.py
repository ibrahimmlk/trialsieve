"""Benchmarks for Tasks 3-5 with the manual Qwen2 implementation. Writes results/benchmarks/*.json.

    python scripts/benchmark.py --part all --adapter outputs/lora-qwen1.5b/final \
        --draft-adapter outputs/lora-qwen0.5b-draft/final

Every timed configuration gets one untimed warm-up run, then --repeats timed runs
(CUDA-synchronised); the median is reported. Prompts come from the validation split only.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialsieve.attention import ATTENTION, reference_attention  # noqa: E402
from trialsieve.data import load_split, messages  # noqa: E402
from trialsieve.decoding import greedy_kv, greedy_kv_batch, greedy_no_cache, speculative  # noqa: E402
from trialsieve.manual_model import ManualQwen2  # noqa: E402

FILLER = ("The patient reports intermittent chest discomfort, fatigue and shortness of breath on exertion. "
          "Past history includes hypertension, hyperlipidaemia and a remote appendectomy. ")


def load(name, adapter, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    return model


def median_runs(fn, repeats):
    fn()  # warm-up (allocator, kernels, caches)
    runs = [fn() for _ in range(repeats)]
    tokens = runs[0][0]
    keys = [k for k, v in runs[0][1].items() if isinstance(v, (int, float)) and v is not None]
    stats = {k: statistics.median(r[1][k] for r in runs) for k in keys}
    stats.update({k: v for k, v in runs[0][1].items() if k not in stats})
    return tokens, stats


def chat_ids(tok, msgs):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


def prompt_of_length(tok, n_tokens):
    text = (FILLER * (n_tokens // 20 + 2))
    ids = tok(text).input_ids[:n_tokens]
    return ids


def bench_kv(model, tok, args):
    """Task 3: uncached vs cached greedy decoding, several prompt/output lengths, fixed length (EOS ignored)."""
    rows = []
    for p_len in args.prompt_lens:
        for n_new in args.new_tokens:
            ids = prompt_of_length(tok, p_len)
            ref, s_ref = median_runs(lambda: greedy_no_cache(model, ids, n_new), args.repeats)
            kv, s_kv = median_runs(lambda: greedy_kv(model, ids, n_new), args.repeats)
            mismatch = next((i for i, (a, b) in enumerate(zip(ref, kv)) if a != b), None)
            rows.append({"prompt_tokens": p_len, "new_tokens": n_new, "tokens_equal": ref == kv,
                         "first_mismatch": mismatch, "no_cache": s_ref, "kv_cache": s_kv,
                         "speedup_total": s_ref["total_s"] / s_kv["total_s"]})
            print(f"[kv] prompt={p_len} new={n_new} equal={ref == kv} no_cache={s_ref['total_s']:.2f}s "
                  f"kv={s_kv['total_s']:.2f}s speedup={rows[-1]['speedup_total']:.2f}x", flush=True)
    return rows


def bench_attention(args, device):
    """Task 4: reference vs tiled vs SDPA on random tensors (Qwen2.5-1.5B head shape: 12 heads x 128)."""
    rows = []
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    for T in args.attn_lens:
        for causal in [True, False]:
            g = torch.Generator(device="cpu").manual_seed(T)
            q, k, v = (torch.randn(1, 12, T, 128, generator=g).to(device, dtype) for _ in range(3))
            exact = reference_attention(q.double(), k.double(), v.double(), causal=causal)
            row = {"seq_len": T, "causal": causal, "dtype": str(dtype)}
            for name, fn in ATTENTION.items():
                def run():
                    if device.type == "cuda":
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated() if device.type == "cuda" else 0
                    t0 = time.perf_counter()
                    out = fn(q, k, v, causal=causal)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    peak = torch.cuda.max_memory_allocated() - base if device.type == "cuda" else None
                    return out, {"ms": dt * 1e3, "peak_mem_bytes": peak}
                out, st = median_runs(run, args.repeats)
                err = (out.double() - exact).abs()
                row[name] = st | {"max_abs_err_vs_fp64": err.max().item(), "mean_abs_err_vs_fp64": err.mean().item()}
            rows.append(row)
            print(f"[attn] T={T} causal={causal} " + " ".join(f"{n}={row[n]['ms']:.2f}ms" for n in ATTENTION), flush=True)
    return rows


def bench_attention_e2e(model, tok, args):
    """Task 4 end to end: greedy outputs with each attention kernel inside the real model."""
    rows = []
    for r in load_split("val")[: args.n_prompts]:
        ids = chat_ids(tok, messages(r))
        outs = {}
        for name in ATTENTION:
            model.set_attention(name)
            outs[name], st = median_runs(lambda: greedy_kv(model, ids, 48, eos_ids=args.eos), 1)
            outs[name + "_s"] = st["total_s"]
        model.set_attention("reference")
        rows.append({"annotation_id": r["annotation_id"], "prompt_tokens": len(ids),
                     "tiled_equals_reference": outs["tiled"] == outs["reference"],
                     "sdpa_equals_reference": outs["sdpa"] == outs["reference"],
                     "seconds": {n: outs[n + "_s"] for n in ATTENTION}})
    print(f"[attn-e2e] tiled==reference on {sum(r['tiled_equals_reference'] for r in rows)}/{len(rows)} prompts", flush=True)
    return rows


def bench_speculative(target, drafts, tok, args):
    """Task 5: greedy target vs speculative decoding on validation prompts + longer free-text prompts."""
    prompts = []
    for r in load_split("val")[: args.n_prompts]:
        prompts.append(("criterion", chat_ids(tok, messages(r)), 64))
    for q in ["Explain in detail how clinical trial inclusion and exclusion criteria are used to screen patients.",
              "Summarise the main risk factors for coronary artery disease and how each is assessed."]:
        prompts.append(("free_text", chat_ids(tok, [{"role": "user", "content": q}]), 128))

    rows = []
    for kind, ids, n_new in prompts:
        base, s_base = median_runs(lambda: greedy_kv(target, ids, n_new, eos_ids=args.eos), args.repeats)
        for draft_name, draft in drafts.items():
            for k in args.spec_k:
                out, s = median_runs(lambda: speculative(target, draft, ids, n_new, eos_ids=args.eos, k=k), args.repeats)
                rows.append({"kind": kind, "prompt_tokens": len(ids), "draft": draft_name, "k": k,
                             "tokens_equal": out == base, "new_tokens": len(out), "greedy": s_base, "speculative": s,
                             "speedup_total": s_base["total_s"] / s["total_s"]})
                print(f"[spec] {kind} draft={draft_name} k={k} equal={out == base} accept={s['acceptance_rate']:.2f} "
                      f"speedup={rows[-1]['speedup_total']:.2f}x", flush=True)
    return rows


def bench_batch(model, tok, args):
    """Screening one patient-trial pair: sequential vs prefix-reuse vs batched ragged KV decoding."""
    from trialsieve.screening import validation_examples
    g = list(validation_examples().values())[0]
    crit = [("inclusion", c) for c in g["inclusion"]] + [("exclusion", c) for c in g["exclusion"]]
    prompts = [chat_ids(tok, messages({"note": g["note"], "criterion_type": t, "criterion_text": c})) for t, c in crit]
    t0 = time.perf_counter()
    seq = [greedy_kv(model, p, 64, eos_ids=args.eos)[0] for p in prompts]
    t_seq = time.perf_counter() - t0
    state, t0 = {}, time.perf_counter()
    reuse = [greedy_kv(model, p, 64, eos_ids=args.eos, prefix_state=state)[0] for p in prompts]
    t_reuse = time.perf_counter() - t0
    batched, st = greedy_kv_batch(model, prompts, 64, eos_ids=args.eos)
    row = {"criteria": len(prompts), "prompt_tokens_mean": sum(map(len, prompts)) / len(prompts),
           "shared_prefix_tokens": st["shared_prefix_tokens"], "sequential_s": t_seq, "prefix_reuse_s": t_reuse,
           "batched_s": st["total_s"], "batched_prefill_s": st["prefill_s"], "batched_decode_s": st["decode_s"],
           "reuse_equal": reuse == seq, "batched_equal": batched == seq}
    print(f"[batch] {row}", flush=True)
    return [row]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["kv", "attention", "speculative", "batch", "all"], default="all")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--draft-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--draft-adapter", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[64, 256, 512])
    ap.add_argument("--new-tokens", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--attn-lens", type=int, nargs="+", default=[128, 512, 1024, 2048])
    ap.add_argument("--spec-k", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "benchmarks"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    args.eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    env = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
           "dtype": str(dtype), "torch": torch.__version__, "repeats": args.repeats, "statistic": "median after 1 warm-up"}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save(name, rows, **extra):
        (out / f"{name}.json").write_text(json.dumps({"env": env, **extra, "rows": rows}, indent=2, default=str))

    if args.part in ("attention", "all"):
        save("attention_kernels", bench_attention(args, device))
    if args.part == "attention" and not args.adapter:
        return
    target = ManualQwen2(load(args.model, args.adapter, dtype, device))
    if args.part in ("kv", "all"):
        save("kv_cache", bench_kv(target, tok, args), model=args.model, adapter=args.adapter)
    if args.part in ("attention", "all"):
        save("attention_e2e", bench_attention_e2e(target, tok, args), model=args.model, adapter=args.adapter)
    if args.part == "batch":
        save("batched_screening", bench_batch(target, tok, args), model=args.model, adapter=args.adapter)
    if args.part in ("speculative", "all"):
        drafts = {"base-0.5b": ManualQwen2(load(args.draft_model, None, dtype, device))}
        if args.draft_adapter:
            drafts["lora-0.5b"] = ManualQwen2(load(args.draft_model, args.draft_adapter, dtype, device))
        save("speculative", bench_speculative(target, drafts, tok, args), model=args.model, adapter=args.adapter,
             draft_model=args.draft_model, draft_adapter=args.draft_adapter)


if __name__ == "__main__":
    main()
