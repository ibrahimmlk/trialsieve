# TrialSieve

Criterion-level clinical-trial eligibility screening with evidence, using a fine-tuned open-weight Transformer
(**Qwen2.5-1.5B-Instruct + LoRA**). The Transformer forward pass, **KV cache**, **tiled online-softmax attention** and
**greedy speculative decoding** (Qwen2.5-0.5B draft) are implemented from scratch. A browser application screens a patient
against a trial and exports an auditable report.

> **Educational screening prototype only. It is not a clinical enrollment decision tool.**

Report: [report/main.pdf](report/main.pdf) (IEEE format, LaTeX source in [report/](report/)).

## Results (held-out test split: 8 patients, 206 criteria, evaluated once)

| Model | Accuracy | Macro-F1 | Evidence micro-F1 | Evidence F1 (gold non-empty) |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | 0.325 | 0.256 | 0.123 | 0.111 |
| **+ LoRA (ours)** | **0.743** | **0.545** | **0.508** | **0.533** |

Inference (all decoders are token-identical to their reference path; see the report for details):

| Mechanism | T4 GPU (fp16) | Laptop CPU i7-1255U (fp32) |
|---|---|---|
| KV cache vs. no cache | up to **4.1x** faster (512-token prompts) | ~6x faster per prompt |
| Tiled online-softmax attention | **13x less memory** than reference at T=2048, 9x slower (unfused) | — |
| Speculative decoding (LoRA 0.5B draft, k=4) | 1.07x (criterion prompts) | **1.17x** criterion, **1.27x** free text |
| Batched decoding, one cache length per row (21 criteria) | — | **4.2x** faster than sequential |

## Quick start: the application (Windows)

1. Install Python 3.11+ and Git.
2. `git clone https://github.com/ibrahimmlk/trialsieve.git`
3. Double-click **`run_app.bat`**. The first run creates `.venv`, installs dependencies and downloads the models
   (about 4.2 GB). The browser then opens at <http://127.0.0.1:7860>. Later runs start offline in about 15 s.

Other platforms: `python -m venv .venv`, activate it, `pip install torch --index-url https://download.pytorch.org/whl/cpu`,
`pip install -r requirements.txt`, then `python app/app.py`.

Application tabs:
1. **Screen a patient.** Paste a note (free text is split into numbered sentences automatically), then type the criteria,
   load a validation example or fetch a trial from ClinicalTrials.gov by NCT ID. For every criterion the app shows the
   label, the cited sentence IDs, the evidence text and an explanation. It also gives a rule-based overall outcome and
   lists missing information. Reports download as HTML or JSON with provenance. You can switch between the fine-tuned
   and base model, the decoding mode and the attention kernel.
2. **Inference lab.** Runs the same prompt through no-cache, KV-cache, speculative and `generate()` decoding live,
   showing token identity, timings, acceptance rate and cache size. Also includes a live attention-kernel check and the
   recorded GPU benchmarks.
3. **Evaluation results.** Metrics, confusion matrices, per-class scores and training curves.

## Models

| | Link |
|---|---|
| Target base | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0) |
| Target LoRA adapter (ours) | [ibrahimmlk/trialsieve-qwen2.5-1.5b-lora](https://huggingface.co/ibrahimmlk/trialsieve-qwen2.5-1.5b-lora) |
| Draft base | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| Draft LoRA adapter (ours) | [ibrahimmlk/trialsieve-qwen2.5-0.5b-draft-lora](https://huggingface.co/ibrahimmlk/trialsieve-qwen2.5-0.5b-draft-lora) |

The app downloads all of them automatically. To fetch them by hand:
`hf download ibrahimmlk/trialsieve-qwen2.5-1.5b-lora`.

## Data and split

Only [ncbi/TrialGPT-Criterion-Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations) is
used (copy in `data/raw/`). The patient-grouped split sorts the unique patient IDs, shuffles them with
`np.random.default_rng(42)` and assigns 37/8/8 to train/val/test:

```
python scripts/make_split.py
```

The script asserts that patients and annotations are disjoint across splits and writes
[data/splits/split_manifest.json](data/splits/split_manifest.json) (patient IDs, counts, label distributions).
GPT-4 prediction and explanation columns are removed from the split files and are never used as inputs.

## Reproducing everything

| Step | Where | Command / notebook |
|---|---|---|
| Split | local | `python scripts/make_split.py` |
| Base vs. LoRA on validation, LoRA training | Kaggle T4 | [notebooks/kaggle_finetune.ipynb](notebooks/kaggle_finetune.ipynb) (`scripts/train.py`, `scripts/evaluate.py`) |
| Tests, draft training, GPU benchmarks, final test | Kaggle T4 | [notebooks/kaggle_benchmarks_final.ipynb](notebooks/kaggle_benchmarks_final.ipynb) |
| CPU benchmarks | local | `python scripts/benchmark.py --part batch --adapter <lora>` and `--part speculative` |
| Error analysis | local | `python scripts/error_analysis.py` |
| Figures | local | `python scripts/make_figures.py` |
| Unit tests | local | `pytest tests` (40 tests: attention kernels; manual model vs. HF; cached, ragged and speculative decoding) |

`evaluate.py` refuses the test split unless `--final` is passed. All logs and metrics are in [results/](results/).
Training config: LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o/gate/up/down; lr 2e-4 cosine; 3 epochs; effective batch 16;
fp16; seed 42; checkpoint selected by validation loss. All evaluation uses greedy decoding with Qwen's default
`repetition_penalty=1.1` overridden to 1.0.

## Repository layout

```
src/trialsieve/
  data.py           prompts, JSON targets, output parsing (no GPT-4 fields)
  metrics.py        accuracy, macro-F1, per-class, confusion; evidence micro/macro with empty-set rules
  manual_model.py   from-scratch Qwen2 forward pass, pre-allocated KV cache, batched decoding step with per-row lengths
  attention.py      reference attention, tiled online-softmax attention, SDPA baseline
  decoding.py       no-cache, KV-cache (+prefix reuse), batched KV, greedy speculative decoding
  screening.py      application engine: validation, aggregation rule, explanations, reports
app/app.py          Gradio application
scripts/            split, training, evaluation, benchmarks, error analysis, figures
tests/              correctness tests
results/            metrics, predictions, training logs, GPU/CPU benchmarks, error analysis
report/             IEEE LaTeX report and figures
```

## AI assistance

Code and report drafting were assisted by an AI coding assistant (Claude Code). Every component is covered by tests, and
every reported number comes from the result files in this repository. See Appendix A of the report.
