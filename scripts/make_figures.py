"""Generate report figures (PDF) from results/*.json into report/figures/."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "font.family": "serif", "figure.dpi": 150})
COL = 3.5  # IEEE column width (inches)


def load(p):
    return json.loads((R / p).read_text())


def training():
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for path, name, c in [("training/log_history.json", "1.5B target", "C0"), ("draft_training/log_history.json", "0.5B draft", "C1")]:
        h = load(path)
        tr = [(x["step"], x["loss"]) for x in h if "loss" in x]
        ev = [(x["step"], x["eval_loss"]) for x in h if "eval_loss" in x]
        ax.plot(*zip(*tr), color=c, lw=1, label=f"{name} train")
        ax.plot(*zip(*ev), "o--", color=c, ms=2.5, lw=1, label=f"{name} validation")
    best = load("training/train_config.json")["best_checkpoint"].split("-")[-1]
    ax.axvline(int(best), color="gray", ls=":", lw=1)
    ax.text(int(best) + 1, ax.get_ylim()[1] * 0.85, f"selected\nstep {best}", fontsize=6, color="gray")
    ax.set_xlabel("optimizer step (effective batch 16)")
    ax.set_ylabel("cross-entropy (answer tokens)")
    ax.set_ylim(0.03, 0.36)
    ax.legend(ncol=2, loc="upper right", fontsize=6)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "training.pdf")


def confusion():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    order = ["included", "not included", "excluded", "not excluded", "not enough information", "not applicable"]
    short = ["inc", "not inc", "exc", "not exc", "NEI", "N/A"]
    for ax, tag, title in zip(axes, ["base", "finetuned"], ["Base model (test)", "LoRA fine-tuned (test)"]):
        cm = load(f"test_{tag}_metrics.json")["all"]["classification"]["confusion_matrix"]
        idx = {l: i for i, l in enumerate(cm["labels"])}
        M = np.zeros((6, 6), int)
        for gi, g in enumerate(order):
            for pi, p in enumerate(order):
                if g in idx and p in idx:
                    M[gi, pi] = cm["matrix"][idx[g]][idx[p]]
        ax.imshow(M, cmap="Blues")
        for i in range(6):
            for j in range(6):
                if M[i, j]:
                    ax.text(j, i, M[i, j], ha="center", va="center", fontsize=7, color="white" if M[i, j] > M.max() / 2 else "black")
        ax.set_xticks(range(6), short, rotation=40, ha="right")
        ax.set_yticks(range(6), short)
        ax.set_xlabel("predicted")
        ax.set_ylabel("expert")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / "confusion.pdf")


def kv():
    rows = load("benchmarks/kv_cache.json")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3))
    for p, c in zip([64, 256, 512], ["C0", "C1", "C2"]):
        rr = [r for r in rows if r["prompt_tokens"] == p]
        n = [r["new_tokens"] for r in rr]
        axes[0].plot(n, [r["no_cache"]["total_s"] for r in rr], "o--", color=c, ms=3, label=f"no cache, prompt {p}")
        axes[0].plot(n, [r["kv_cache"]["total_s"] for r in rr], "s-", color=c, ms=3, label=f"KV cache, prompt {p}")
    axes[0].set_xlabel("generated tokens")
    axes[0].set_ylabel("end-to-end latency (s)")
    axes[0].legend(ncol=1, fontsize=6)
    axes[0].grid(alpha=.3)
    for key, mk, lab in [("no_cache", "o--", "no cache"), ("kv_cache", "s-", "KV cache")]:
        pl = [64, 256, 512]
        axes[1].plot(pl, [np.mean([r[key]["decode_tok_per_s"] for r in rows if r["prompt_tokens"] == p]) for p in pl], mk, ms=3, label=lab)
    axes[1].set_xlabel("prompt length (tokens)")
    axes[1].set_ylabel("decode throughput (tokens/s)")
    axes[1].legend()
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "kv_cache.pdf")


def attention():
    rows = [r for r in load("benchmarks/attention_kernels.json")["rows"] if r["causal"]]
    T = [r["seq_len"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.2))
    for name, mk in [("reference", "o-"), ("tiled", "s-"), ("sdpa", "^-")]:
        axes[0].plot(T, [r[name]["ms"] for r in rows], mk, ms=3, label=name)
        axes[1].plot(T, [r[name]["peak_mem_bytes"] / 2**20 for r in rows], mk, ms=3, label=name)
    for ax, yl in zip(axes, ["time (ms)", "peak extra memory (MiB)"]):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length T (12 heads x 128, causal, fp16, T4)")
        ax.set_ylabel(yl)
        ax.grid(alpha=.3, which="both")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "attention.pdf")


def speculative():
    rows = load("benchmarks/speculative.json")["rows"]
    agg = defaultdict(list)
    for r in rows:
        agg[(r["kind"], r["draft"], r["k"])].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.2))
    for kind, ls in [("criterion", "-"), ("free_text", "--")]:
        for draft, c in [("base-0.5b", "C0"), ("lora-0.5b", "C3")]:
            ks = sorted({k for (kd, d, k) in agg if kd == kind and d == draft})
            acc = [np.mean([x["speculative"]["acceptance_rate"] for x in agg[(kind, draft, k)]]) for k in ks]
            sp = [np.mean([x["speedup_total"] for x in agg[(kind, draft, k)]]) for k in ks]
            lab = f"{draft} draft, {'criterion' if kind == 'criterion' else 'free text'}"
            axes[0].plot(ks, acc, "o" + ls, color=c, ms=3, label=lab)
            axes[1].plot(ks, sp, "o" + ls, color=c, ms=3, label=lab)
    axes[1].axhline(1, color="gray", lw=.8)
    axes[0].set_ylabel("draft acceptance rate")
    axes[1].set_ylabel("speed-up vs greedy KV (T4)")
    for ax in axes:
        ax.set_xlabel("draft length k")
        ax.set_xticks([2, 4, 6])
        ax.grid(alpha=.3)
        ax.legend(fontsize=5.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "speculative.pdf")


if __name__ == "__main__":
    for f in [training, confusion, kv, attention, speculative]:
        f()
        print("wrote", f.__name__)
