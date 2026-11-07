"""Rule-based error taxonomy over test predictions; writes results/error_analysis.{json,md}.

    python scripts/error_analysis.py --split test --tag finetuned
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trialsieve.data import load_split, note_sentences  # noqa: E402

NEGATIVE = {"not excluded", "not included"}
DECIDED = {"included", "not included", "excluded", "not excluded"}

# Ordered: each prediction gets the first matching category.
CATEGORIES = OrderedDict([
    ("absence_as_negative", ("Missing information read as a negative finding",
                             lambda p, g, e: g["label"] == "not enough information" and p["label"] in NEGATIVE)),
    ("overcommit", ("Decided a criterion the experts judged undecidable",
                    lambda p, g, e: g["label"] == "not enough information" and p["label"] in DECIDED - NEGATIVE)),
    ("contradictory_evidence", ("Cited evidence yet answered 'not enough information'",
                                lambda p, g, e: p["label"] == "not enough information" and g["label"] in DECIDED and p["evidence"])),
    ("underconfident", ("Answered 'not enough information' although the note decides the criterion",
                        lambda p, g, e: p["label"] == "not enough information" and g["label"] in DECIDED)),
    ("missed_not_applicable", ("Missed a 'not applicable' criterion",
                               lambda p, g, e: g["label"] == "not applicable" and p["label"] != "not applicable")),
    ("wrong_polarity", ("Opposite decision (met vs not met)",
                        lambda p, g, e: g["label"] in DECIDED and p["label"] in DECIDED and p["label"] != g["label"])),
    ("other_label_error", ("Other label error", lambda p, g, e: p["label"] != g["label"])),
    ("correct_label_unsupported", ("Correct label but no evidence cited where experts cited some",
                                   lambda p, g, e: g["evidence"] and not p["evidence"])),
    ("correct_label_spurious_evidence", ("Correct label but cited sentences the experts did not",
                                         lambda p, g, e: set(p["evidence"]) - set(g["evidence"]))),
    ("correct_label_incomplete_evidence", ("Correct label but missed some expert evidence sentences",
                                           lambda p, g, e: set(g["evidence"]) - set(p["evidence"]))),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="finetuned")
    ap.add_argument("--per-category", type=int, default=2)
    args = ap.parse_args()

    notes = {r["annotation_id"]: r["note"] for r in load_split(args.split)}
    preds = [json.loads(l) for l in open(ROOT / "results" / f"{args.split}_{args.tag}_predictions.jsonl", encoding="utf-8")]
    buckets = {k: [] for k in CATEGORIES}
    for p in preds:
        g = {"label": p["gold_label"], "evidence": p["gold_evidence"]}
        for key, (_, rule) in CATEGORIES.items():
            if rule(p, g, None):
                buckets[key].append(p)
                break

    out = {"split": args.split, "tag": args.tag, "n": len(preds), "categories": {}}
    md = [f"# Error analysis ({args.split}, {args.tag}, n={len(preds)})\n"]
    for key, (title, _) in CATEGORIES.items():
        items = sorted(buckets[key], key=lambda p: p["annotation_id"])
        examples = []
        for p in items[: args.per_category]:
            sents = dict(note_sentences(notes[p["annotation_id"]]))
            examples.append({**{k: p[k] for k in ["annotation_id", "patient_id", "trial_id", "criterion_type", "criterion_text",
                                                  "gold_label", "label", "gold_evidence", "evidence", "raw_output"]},
                             "gold_evidence_text": {i: sents.get(i) for i in p["gold_evidence"]},
                             "pred_evidence_text": {i: sents.get(i) for i in p["evidence"]}})
        out["categories"][key] = {"title": title, "count": len(items), "examples": examples}
        md.append(f"## {title} — {len(items)} cases\n")
        for e in examples:
            md.append(f"- **#{e['annotation_id']}** ({e['patient_id']}, {e['trial_id']}, {e['criterion_type']}): "
                      f"*{e['criterion_text']}*  \n  expert **{e['gold_label']}** {e['gold_evidence']} · predicted **{e['label']}** "
                      f"{e['evidence']}  \n  expert evidence: {e['gold_evidence_text'] or '—'}  \n  predicted evidence: "
                      f"{e['pred_evidence_text'] or '—'}\n")
    (ROOT / "results" / "error_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "results" / "error_analysis.md").write_text("\n".join(md), encoding="utf-8")
    for key, v in out["categories"].items():
        print(f"{v['count']:4d}  {v['title']}")


if __name__ == "__main__":
    main()
