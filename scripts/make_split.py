"""Create the mandated patient-grouped 37/8/8 split and its manifest.

Protocol (from the assignment): sort unique patient IDs, shuffle with
numpy.random.default_rng(42), first 37 -> train, next 8 -> val, last 8 -> test.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "trialgpt_criterion_annotations.parquet"
OUT = ROOT / "data" / "splits"
SEED = 42
SIZES = {"train": 37, "val": 8, "test": 8}

# GPT-4 outputs must never be model inputs; they are dropped from the split files.
GPT4_COLUMNS = ["gpt4_explanation", "explanation_correctness", "gpt4_sentences", "gpt4_eligibility"]


def main():
    df = pd.read_parquet(RAW)
    ids = np.array(sorted(df["patient_id"].unique()))
    assert len(ids) == sum(SIZES.values()), f"expected 53 patients, got {len(ids)}"

    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)

    bounds, start = {}, 0
    for name, n in SIZES.items():
        bounds[name] = ids[start:start + n].tolist()
        start += n

    # Leakage checks: every patient in exactly one split, every annotation in exactly one split.
    sets = {k: set(v) for k, v in bounds.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert set().union(*sets.values()) == set(ids)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "protocol": "sorted unique patient_id -> np.random.default_rng(42).shuffle -> 37/8/8",
                "source": "ncbi/TrialGPT-Criterion-Annotations", "splits": {}}
    seen = set()
    for name, pids in bounds.items():
        part = df[df["patient_id"].isin(pids)].drop(columns=GPT4_COLUMNS)
        assert not (seen & set(part["annotation_id"]))
        seen |= set(part["annotation_id"])
        part.to_json(OUT / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
        manifest["splits"][name] = {
            "patient_ids": pids,
            "n_patients": len(pids),
            "n_annotations": len(part),
            "label_counts": part["expert_eligibility"].value_counts().to_dict(),
            "criterion_type_counts": part["criterion_type"].value_counts().to_dict(),
        }
    assert len(seen) == len(df)

    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    for name, info in manifest["splits"].items():
        print(f"{name:5s} patients={info['n_patients']:2d} annotations={info['n_annotations']:4d} {info['label_counts']}")
    print("Leakage checks passed.")


if __name__ == "__main__":
    main()
