"""Criterion classification and evidence-sentence metrics.

Evidence conventions (also stated in the report):
- micro: TP/FP/FN pooled over all records.
- macro: per-record P/R/F1 averaged. Gold and prediction both empty -> P=R=F1=1
  (correctly cited nothing). Gold empty, prediction non-empty -> P=0, R=1, F1=0.
  Gold non-empty, prediction empty -> P=1, R=0, F1=0.
- macro_nonempty_gold: the same average restricted to records with annotated evidence.
"""
from collections import Counter

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .data import ALL_LABELS, INVALID


def classification_report(gold, pred):
    labels = [l for l in ALL_LABELS if l in set(gold) | set(pred)]
    if INVALID in pred:
        labels.append(INVALID)
    present = [l for l in labels if l in set(gold)]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=present, zero_division=0)
    return {
        "n": len(gold),
        "accuracy": accuracy_score(gold, pred),
        # Macro-F1 over classes that occur in the gold labels of this subset.
        "macro_f1": f1_score(gold, pred, labels=present, average="macro", zero_division=0),
        "per_class": {l: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])} for i, l in enumerate(present)},
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(gold, pred, labels=labels).tolist()},
        "invalid_outputs": sum(x == INVALID for x in pred),
        "pred_distribution": dict(Counter(pred)),
    }


def _prf(g, p):
    g, p = set(g), set(p)
    tp = len(g & p)
    if not g and not p:
        return 1.0, 1.0, 1.0
    prec = tp / len(p) if p else 1.0
    rec = tp / len(g) if g else 1.0
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return prec, rec, f1


def evidence_report(gold_sets, pred_sets):
    tp = fp = fn = 0
    per = []
    for g, p in zip(gold_sets, pred_sets):
        g, p = set(g), set(p)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        per.append((bool(g), _prf(g, p)))
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0

    def avg(rows):
        if not rows:
            return {"precision": None, "recall": None, "f1": None, "n": 0}
        return {k: sum(x[i] for x in rows) / len(rows) for i, k in enumerate(["precision", "recall", "f1"])} | {"n": len(rows)}

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f, "tp": tp, "fp": fp, "fn": fn},
        "macro": avg([s for _, s in per]),
        "macro_nonempty_gold": avg([s for nonempty, s in per if nonempty]),
        "empty_gold_records": sum(not ne for ne, _ in per),
        "empty_gold_and_empty_pred": sum(not g and not p for g, p in zip(gold_sets, pred_sets)),
    }


def full_report(records, preds):
    """records: split rows; preds: dicts with 'label' and 'evidence'."""
    out = {}
    for subset in ["all", "inclusion", "exclusion"]:
        idx = [i for i, r in enumerate(records) if subset == "all" or r["criterion_type"] == subset]
        if not idx:
            continue
        gold = [records[i]["expert_eligibility"] for i in idx]
        pred = [preds[i]["label"] for i in idx]
        out[subset] = {
            "classification": classification_report(gold, pred),
            "evidence": evidence_report([preds[i]["gold_evidence"] for i in idx], [preds[i]["evidence"] for i in idx]),
        }
    return out
