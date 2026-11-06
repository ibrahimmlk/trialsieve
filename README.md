# TrialSieve

Criterion-level clinical-trial eligibility screening with a fine-tuned open-weight Transformer
(Qwen2.5-1.5B-Instruct + LoRA), plus from-scratch KV-cached decoding, tiled online-softmax
attention and greedy speculative decoding (Qwen2.5-0.5B-Instruct draft).

> Educational screening prototype only — not a clinical enrollment decision tool.

## Data split
Dataset: [ncbi/TrialGPT-Criterion-Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations).
Patient-grouped split (sorted IDs, `np.random.default_rng(42)` shuffle, 37/8/8):

```
python scripts/make_split.py
```
Manifest: [data/splits/split_manifest.json](data/splits/split_manifest.json). GPT-4 prediction columns are dropped from all split files.
