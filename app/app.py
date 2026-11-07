"""TrialSieve browser application (Gradio). Start with run_app.bat or: python app/app.py"""
import html
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _all_cached():
    from huggingface_hub import try_to_load_from_cache
    files = [("Qwen/Qwen2.5-1.5B-Instruct", "model.safetensors"), ("Qwen/Qwen2.5-0.5B-Instruct", "model.safetensors"),
             ("ibrahimmlk/trialsieve-qwen2.5-1.5b-lora", "adapter_model.safetensors"),
             ("ibrahimmlk/trialsieve-qwen2.5-0.5b-draft-lora", "adapter_model.safetensors")]
    return all(isinstance(try_to_load_from_cache(r, f), str) for r, f in files)


# Work offline when every model file is already downloaded (the demo must not depend on Wi-Fi).
if _all_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from trialsieve.attention import ATTENTION, reference_attention  # noqa: E402
from trialsieve.data import number_note  # noqa: E402
from trialsieve.screening import (DISCLAIMER, MODES, Engine, InputError, fetch_trial_criteria,  # noqa: E402
                                  parse_criteria_block, report_html, validation_examples)

RESULTS = ROOT / "results"
REPORTS = ROOT / "outputs" / "reports"
EXAMPLES = validation_examples()
ATTN_CHOICES = {"Reference (explicit softmax)": "reference", "Tiled online-softmax": "tiled", "PyTorch SDPA (fused)": "sdpa"}
LAB_MODES = {"No cache (reference)": "no_cache", "KV cache (manual)": "kv_cache",
             "Speculative (0.5B LoRA draft)": "speculative", "HF generate (library)": "hf_generate"}

print("Loading models (first start can take ~1 minute)...", flush=True)
ENGINE = Engine()
print(f"Target model ready in {ENGINE.load_seconds:.0f}s. Offline mode: {os.environ.get('HF_HUB_OFFLINE') == '1'}", flush=True)


# ---------------------------------------------------------------- Screening tab
def load_example(key):
    if not key:
        raise gr.Error("Pick an example first.")
    g = EXAMPLES[key]
    return g["note"], "\n".join(g["inclusion"]), "\n".join(g["exclusion"]), g["gold"], f"Loaded {key} — {g['title']}"


def fetch_trial(nct):
    try:
        title, inc, exc = fetch_trial_criteria(nct)
    except InputError as e:
        raise gr.Error(str(e))
    if not inc and not exc:
        raise gr.Error("The trial record has no parsable eligibility criteria. Enter them manually.")
    return "\n".join(inc), "\n".join(exc), {}, f"Fetched {nct.strip().upper()}: {title} ({len(inc)} inclusion, {len(exc)} exclusion)"


def verdict_html(rep, gold_hits):
    s = rep["summary"]
    color = {"Likely ineligible": "#c62828", "Potentially eligible": "#2e7d32"}.get(s["verdict"], "#b26a00")
    missing = "".join(f"<li>{html.escape(m)}</li>" for m in s["missing_information"][:12])
    gold = f"<p><b>Agreement with expert labels (validation example):</b> {gold_hits}</p>" if gold_hits else ""
    p = rep["provenance"]
    return f"""
<div style="border-left:6px solid {color};padding:10px 14px;border-radius:6px;background:rgba(127,127,127,.08)">
<div style="font-size:1.4em;font-weight:700;color:{color}">Preliminary outcome: {s['verdict']}</div>
<p>{s['reason']}</p><p style="font-size:.9em"><i>Rule:</i> {s['rule']}</p>
{'<p><b>Missing information:</b></p><ul>' + missing + '</ul>' if missing else ''}{gold}
<p style="font-size:.85em">Runtime {p['runtime_s']} s · note {p['note_tokens']} tokens · mode {rep['settings']['decoding']} ·
attention {rep['settings']['attention']} · model {'fine-tuned (LoRA)' if rep['settings']['fine_tuned_adapter'] else 'base (no fine-tuning)'}</p>
</div>
<p style="background:#fff4e5;color:#5a3a00;border:1px solid #f0a020;padding:8px;border-radius:6px"><b>Notice:</b> {DISCLAIMER}</p>"""


def run_screening(note, inc_text, exc_text, gold, model_choice, mode_label, attn_label, k, progress=gr.Progress()):
    criteria = [("inclusion", c) for c in parse_criteria_block(inc_text)] + \
               [("exclusion", c) for c in parse_criteria_block(exc_text)]
    try:
        rep = ENGINE.screen(note, criteria, mode=MODES[mode_label], attention=ATTN_CHOICES[attn_label], k=int(k),
                            use_adapter=model_choice.startswith("Fine-tuned"), progress=progress)
    except InputError as e:
        raise gr.Error(str(e))
    gold = gold or {}
    rows, hits, n_gold = [], 0, 0
    for r in rep["results"]:
        expert = gold.get((r["type"], r["criterion"].strip()), "")
        if expert:
            n_gold += 1
            hits += expert == r["label"]
        rows.append([r["index"], r["type"], r["criterion"], r["label"], expert,
                     ", ".join(map(str, r["evidence_ids"])) or "—",
                     " | ".join(f"[{i}] {s}" for i, s in r["evidence"]) or "—", r["explanation"]])
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    html_path, json_path = REPORTS / f"screening_{stamp}.html", REPORTS / f"screening_{stamp}.json"
    html_path.write_text(report_html(rep), encoding="utf-8")
    json_path.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    note_html = "<ol start='0' style='margin:0'>" + "".join(
        f"<li value='{i}'>{html.escape(s)}</li>" for i, s in rep["sentences"].items()) + "</ol>"
    gold_hits = f"{hits}/{n_gold} criteria" if n_gold else ""
    return verdict_html(rep, gold_hits), rows, note_html, [str(html_path), str(json_path)]


# ---------------------------------------------------------------- Inference lab tab
LAB_RECORDS = {f"{r['patient_id']} | {r['criterion_type']}: {(r['criterion_text'] or '')[:70]}": r
               for r in __import__("trialsieve.data", fromlist=["load_split"]).load_split("val")[:40]}


def run_lab(source, custom, modes, attn_label, k, max_new):
    from trialsieve.data import messages
    if custom and custom.strip():
        chat = [{"role": "user", "content": custom.strip()}]
        text = ENGINE.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    elif source:
        text = ENGINE.tok.apply_chat_template(messages(LAB_RECORDS[source]), tokenize=False, add_generation_prompt=True)
    else:
        raise gr.Error("Choose a validation prompt or type a custom prompt.")
    if not modes:
        raise gr.Error("Select at least one decoding mode.")
    ids = ENGINE.tok(text, add_special_tokens=False).input_ids
    if len(ids) > 1500:
        raise gr.Error(f"Prompt is {len(ids)} tokens; the lab limit is 1500.")
    rows, outputs, ref = [], [], None
    order = sorted(modes, key=lambda m: list(LAB_MODES).index(m))
    with ENGINE.lock:
        for label in order:
            out, st = ENGINE.decode(ids, LAB_MODES[label], ATTN_CHOICES[attn_label], int(k), True, False, int(max_new))
            if ref is None:
                ref = out
            same = "reference" if out is ref else ("yes" if out == ref else
                   f"no (first diff at token {next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), min(len(out), len(ref)))})")
            fmt = lambda v, d=3: "—" if v is None else (round(v, d) if isinstance(v, float) else v)
            rows.append([label, same, st.get("new_tokens"), fmt(st.get("prefill_s")), fmt(st.get("decode_s")),
                         fmt(st.get("decode_tok_per_s"), 2), fmt(st.get("total_s")),
                         fmt(st.get("acceptance_rate"), 2), fmt(st.get("tokens_per_target_forward"), 2),
                         fmt(st["cache_bytes"] / 2**20, 2) if st.get("cache_bytes") else "—"])
            outputs.append(f"**{label}:** `{ENGINE.tok.decode(out, skip_special_tokens=True)}`")
    info = f"Prompt: {len(ids)} tokens · greedy decoding · fp32 on CPU · identical tokens are checked against the first selected mode."
    return rows, info + "\n\n" + "\n\n".join(outputs)


def run_attention_check(seq_len, causal, block):
    T, rows = int(seq_len), []
    g = torch.Generator().manual_seed(T)
    q, k, v = (torch.randn(1, 12, T, 128, generator=g) for _ in range(3))
    exact = reference_attention(q.double(), k.double(), v.double(), causal=causal)
    for label, name in ATTN_CHOICES.items():
        fn = ATTENTION[name]
        kwargs = {"block_q": int(block), "block_k": int(block)} if name == "tiled" else {}
        fn(q, k, v, causal=causal, **kwargs)  # warm-up
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            out = fn(q, k, v, causal=causal, **kwargs)
            times.append((time.perf_counter() - t0) * 1e3)
        err = (out.double() - exact).abs()
        score_mb = T * T * 12 * 4 / 2**20 if name == "reference" else (int(block) ** 2 * 12 * 4 / 2**20 if name == "tiled" else None)
        rows.append([label, round(sorted(times)[1], 2), f"{err.max().item():.2e}", f"{err.mean().item():.2e}",
                     "—" if score_mb is None else round(score_mb, 2)])
    return rows


# ---------------------------------------------------------------- Results tab (files written by the experiments)
def _load(path):
    try:
        return json.loads((RESULTS / path).read_text())
    except Exception:
        return None


def metrics_table():
    rows = []
    for split in ["val", "test"]:
        for tag in ["base", "finetuned"]:
            m = _load(f"{split}_{tag}_metrics.json")
            if not m:
                continue
            for sub in ["all", "inclusion", "exclusion"]:
                c, e = m[sub]["classification"], m[sub]["evidence"]
                rows.append([split, tag, sub, c["n"], round(c["accuracy"], 3), round(c["macro_f1"], 3),
                             round(e["micro"]["f1"], 3), round(e["macro"]["f1"], 3),
                             None if e["macro_nonempty_gold"]["f1"] is None else round(e["macro_nonempty_gold"]["f1"], 3),
                             c["invalid_outputs"]])
    return rows


def confusion(split, tag):
    m = _load(f"{split}_{tag}_metrics.json")
    if not m:
        return gr.update(value=[[f"No {split} results yet"]], headers=["info"]), []
    cm = m["all"]["classification"]["confusion_matrix"]
    per = m["all"]["classification"]["per_class"]
    return (gr.update(value=[[lab] + row for lab, row in zip(cm["labels"], cm["matrix"])],
                      headers=["expert / predicted"] + cm["labels"]),
            [[k, round(v["precision"], 3), round(v["recall"], 3), round(v["f1"], 3), v["support"]] for k, v in per.items()])


def training_plot():
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for path, name in [("training/log_history.json", "1.5B target"), ("draft_training/log_history.json", "0.5B draft")]:
        h = _load(path) or []
        tr = [(x["step"], x["loss"]) for x in h if "loss" in x]
        ev = [(x["step"], x["eval_loss"]) for x in h if "eval_loss" in x]
        if tr:
            ax.plot(*zip(*tr), label=f"{name} train")
            ax.plot(*zip(*ev), "o--", label=f"{name} validation")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("loss"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def gpu_tables():
    kv = _load("benchmarks/kv_cache.json")
    kv_rows = [[r["prompt_tokens"], r["new_tokens"], r["tokens_equal"], r["no_cache"]["total_s"], r["kv_cache"]["total_s"],
                round(r["no_cache"]["decode_tok_per_s"], 1), round(r["kv_cache"]["decode_tok_per_s"], 1),
                round(r["no_cache"]["peak_mem_bytes"] / 2**20, 1), round(r["kv_cache"]["peak_mem_bytes"] / 2**20, 1),
                round(r["speedup_total"], 2)] for r in (kv or {}).get("rows", [])]
    at = _load("benchmarks/attention_kernels.json")
    at_rows = [[r["seq_len"], r["causal"]] + [round(r[n]["ms"], 2) for n in ATTENTION] +
               [round(r[n]["peak_mem_bytes"] / 2**20, 1) for n in ATTENTION] + [f"{r['tiled']['max_abs_err_vs_fp64']:.1e}"]
               for r in (at or {}).get("rows", [])]
    sp = _load("benchmarks/speculative.json")
    agg = {}
    for r in (sp or {}).get("rows", []):
        a = agg.setdefault((r["kind"], r["draft"], r["k"]), [])
        a.append(r)
    sp_rows = [[kind, draft, k, all(x["tokens_equal"] for x in v),
                round(sum(x["speculative"]["acceptance_rate"] for x in v) / len(v), 2),
                round(sum(x["speculative"]["tokens_per_target_forward"] for x in v) / len(v), 2),
                round(sum(x["speedup_total"] for x in v) / len(v), 2), len(v)] for (kind, draft, k), v in agg.items()]
    env = (kv or {}).get("env", {})
    return kv_rows, at_rows, sp_rows, f"Measured on {env.get('device', '?')} ({env.get('dtype', '?')}), {env.get('statistic', '')}."


# ---------------------------------------------------------------- Layout
with gr.Blocks(title="TrialSieve") as demo:
    gr.Markdown("# TrialSieve — criterion-level clinical-trial eligibility screening\n"
                f"Qwen2.5-1.5B-Instruct + LoRA, decoded with a from-scratch KV-cached Transformer. **{DISCLAIMER}**")
    with gr.Tab("1 · Screen a patient"):
        gold_state = gr.State({})
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row():
                    ex = gr.Dropdown(list(EXAMPLES), label="Validation example (patient | trial)", scale=4)
                    ex_btn = gr.Button("Load example", scale=1)
                with gr.Row():
                    nct = gr.Textbox(label="…or fetch criteria from ClinicalTrials.gov (needs internet)", placeholder="NCT01397994", scale=4)
                    nct_btn = gr.Button("Fetch trial", scale=1)
                status = gr.Markdown()
                note = gr.Textbox(label="Patient note (free text is split into numbered sentences automatically)", lines=9)
                inc = gr.Textbox(label="Inclusion criteria (one per line)", lines=5)
                exc = gr.Textbox(label="Exclusion criteria (one per line)", lines=5)
                with gr.Accordion("Model and inference settings", open=False):
                    model_choice = gr.Radio(["Fine-tuned (LoRA)", "Base model (before fine-tuning)"], value="Fine-tuned (LoRA)", label="Model")
                    mode = gr.Dropdown(list(MODES), value=list(MODES)[0], label="Decoding mode")
                    attn = gr.Radio(list(ATTN_CHOICES), value=list(ATTN_CHOICES)[0], label="Attention kernel")
                    k = gr.Slider(1, 8, value=4, step=1, label="Speculative draft length k")
                run = gr.Button("Screen patient", variant="primary")
            with gr.Column(scale=7):
                verdict = gr.HTML()
                table = gr.Dataframe(headers=["#", "Type", "Criterion", "Predicted", "Expert label", "Evidence IDs",
                                              "Evidence text", "Explanation"], wrap=True, interactive=False)
                with gr.Accordion("Patient note with sentence IDs", open=False):
                    note_view = gr.HTML()
                files = gr.File(label="Download report (HTML + JSON with provenance)", file_count="multiple")
        ex_btn.click(load_example, ex, [note, inc, exc, gold_state, status])
        nct_btn.click(fetch_trial, nct, [inc, exc, gold_state, status])
        note.input(lambda: {}, None, gold_state)  # edited input: expert labels no longer apply
        run.click(run_screening, [note, inc, exc, gold_state, model_choice, mode, attn, k], [verdict, table, note_view, files])

    with gr.Tab("2 · Inference lab"):
        gr.Markdown("Run the **same prompt** through each decoding path live (CPU, fp32, greedy). The first selected mode is "
                    "the reference for the token-identity check.")
        with gr.Row():
            src = gr.Dropdown(list(LAB_RECORDS), value=list(LAB_RECORDS)[0], label="Validation criterion prompt")
            custom = gr.Textbox(label="…or a custom prompt (overrides)", lines=2)
        with gr.Row():
            lab_modes = gr.CheckboxGroup(list(LAB_MODES), value=list(LAB_MODES)[1:], label="Decoding modes "
                                         "(no-cache is ~10x slower on CPU: use <= 16 new tokens)")
            lab_attn = gr.Radio(list(ATTN_CHOICES), value=list(ATTN_CHOICES)[0], label="Attention kernel")
        with gr.Row():
            lab_k = gr.Slider(1, 8, value=4, step=1, label="Speculative k")
            lab_new = gr.Slider(8, 128, value=24, step=8, label="Max new tokens")
        lab_btn = gr.Button("Run comparison", variant="primary")
        lab_table = gr.Dataframe(headers=["Mode", "Tokens identical", "New tokens", "Prefill s", "Decode s", "Decode tok/s",
                                          "Total s", "Draft acceptance", "Tokens / target pass", "KV cache MiB"], interactive=False)
        lab_out = gr.Markdown()
        lab_btn.click(run_lab, [src, custom, lab_modes, lab_attn, lab_k, lab_new], [lab_table, lab_out])

        gr.Markdown("### Attention kernels: live correctness and speed check (random 12-head × 128-dim tensors)")
        with gr.Row():
            a_len = gr.Slider(16, 1024, value=256, step=16, label="Sequence length")
            a_causal = gr.Checkbox(value=True, label="Causal mask")
            a_block = gr.Slider(16, 256, value=64, step=16, label="Tile size (tiled kernel)")
        a_btn = gr.Button("Run attention check")
        a_table = gr.Dataframe(headers=["Kernel", "Median ms", "Max |err| vs fp64", "Mean |err| vs fp64",
                                        "Score matrix held at once (MiB)"], interactive=False)
        a_btn.click(run_attention_check, [a_len, a_causal, a_block], a_table)

        gr.Markdown("### GPU benchmarks (Kaggle Tesla T4, recorded by scripts/benchmark.py)")
        kv_rows, at_rows, sp_rows, env_note = gpu_tables()
        gr.Markdown(env_note)
        gr.Dataframe(kv_rows, headers=["Prompt tok", "New tok", "Tokens equal", "No-cache s", "KV s", "No-cache tok/s",
                                       "KV tok/s", "No-cache MiB", "KV MiB", "Speed-up"], label="KV cache vs no cache")
        gr.Dataframe(at_rows, headers=["T", "Causal", "Ref ms", "Tiled ms", "SDPA ms", "Ref MiB", "Tiled MiB", "SDPA MiB",
                                       "Tiled max err"], label="Attention kernels")
        gr.Dataframe(sp_rows, headers=["Prompt", "Draft", "k", "All tokens equal", "Acceptance", "Tokens/target pass",
                                       "Speed-up", "Runs"], label="Speculative decoding (mean over prompts)")

    with gr.Tab("3 · Evaluation results"):
        gr.Markdown("Criterion-level metrics on the patient-grouped split (validation: 8 patients / 206 criteria; test: 8 / 206).")
        gr.Dataframe(metrics_table(), headers=["Split", "Model", "Subset", "N", "Accuracy", "Macro-F1", "Evidence micro-F1",
                                               "Evidence macro-F1", "Evidence F1 (gold non-empty)", "Invalid"])
        with gr.Row():
            split_sel = gr.Radio(["val", "test"], value="val", label="Split")
            tag_sel = gr.Radio(["finetuned", "base"], value="finetuned", label="Model")
        cm_table = gr.Dataframe(label="Confusion matrix (rows = expert, columns = predicted)")
        pc_table = gr.Dataframe(headers=["Class", "Precision", "Recall", "F1", "Support"], label="Per-class metrics")
        for c in (split_sel, tag_sel):
            c.change(confusion, [split_sel, tag_sel], [cm_table, pc_table])
        gr.Plot(training_plot(), label="Training curves")
        demo.load(lambda: confusion("val", "finetuned"), None, [cm_table, pc_table])

    with gr.Tab("4 · About"):
        gr.Markdown(f"""
**What it does.** For each criterion the model returns one of the dataset's label categories (inclusion: included / not included /
not enough information / not applicable; exclusion: excluded / not excluded / not enough information / not applicable) and the IDs of
the note sentences that support it. The explanation is built only from the cited sentences. The overall outcome is a rule-based
aggregation, not a model prediction, because the dataset has no trial-level ground truth.

**Models.** Target `{ENGINE.names['target']}` + LoRA adapter `{ENGINE.names['adapter']}`; draft `{ENGINE.names['draft']}` +
`ibrahimmlk/trialsieve-qwen2.5-0.5b-draft-lora`. All decoding modes except "HF generate" use the project's own Transformer forward pass,
KV cache and attention kernels (`src/trialsieve`).

**Limits.** Notes up to 1200 tokens, up to 40 criteria per run, 64 generated tokens per criterion, CPU fp32.

**{DISCLAIMER}**""")

if __name__ == "__main__":
    demo.launch(inbrowser=os.environ.get("TRIALSIEVE_NO_BROWSER") != "1", server_name="127.0.0.1", server_port=7860)
