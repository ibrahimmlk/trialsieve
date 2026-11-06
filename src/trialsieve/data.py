"""Prompt construction, training targets and output parsing.

Only patient note, criterion type and criterion text are model inputs. The
GPT-4 columns of the source dataset are dropped by scripts/make_split.py and
never referenced here.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "splits"

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}
ALL_LABELS = ["included", "not included", "excluded", "not excluded", "not enough information", "not applicable"]
INVALID = "invalid output"

DEFINITIONS = {
    "inclusion": (
        "- included: the note shows the patient meets the criterion.\n"
        "- not included: the note shows the patient does not meet the criterion.\n"
        "- not enough information: the note lacks the information needed to decide.\n"
        "- not applicable: the criterion does not apply to this patient."
    ),
    "exclusion": (
        "- excluded: the note shows the patient meets the exclusion criterion.\n"
        "- not excluded: the note shows the patient does not meet the exclusion criterion.\n"
        "- not enough information: the note lacks the information needed to decide.\n"
        "- not applicable: the criterion does not apply to this patient."
    ),
}

SYSTEM_PROMPT = (
    "You are a clinical-trial eligibility screening assistant. Judge one trial criterion "
    "against a patient note using only information stated in the note. Absence of "
    "information is not evidence. Answer with a single JSON object and nothing else."
)


def load_split(name):
    path = SPLIT_DIR / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def note_sentences(note):
    """Return [(id, text)] from a note whose lines look like '3. text'."""
    out = []
    for line in note.split("\n"):
        m = re.match(r"\s*(\d+)\.\s+(.*)", line)
        if m:
            out.append((int(m.group(1)), m.group(2).strip()))
    return out


def number_note(text):
    """Split free text into sentences and number them like the dataset notes (for unseen app inputs)."""
    if note_sentences(text) and all(line.strip() == "" or re.match(r"\s*\d+\.\s", line) for line in text.split("\n")):
        return text.strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", " ".join(text.split()))
    return "\n".join(f"{i}. {p}" for i, p in enumerate(p for p in parts if p))


def user_prompt(note, criterion_type, criterion_text):
    criterion_type = criterion_type.strip().lower()
    labels = LABELS[criterion_type]
    return (
        "Patient note (each sentence starts with its sentence ID):\n"
        f"{note.strip()}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {(criterion_text or '').strip()}\n\n"
        f"Allowed labels:\n{DEFINITIONS[criterion_type]}\n\n"
        'Respond with JSON: {"label": "<one of: ' + " | ".join(labels) + '>", '
        '"evidence_sentence_ids": [<IDs of note sentences that support the label>]}'
    )


def messages(record):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(record["note"], record["criterion_type"], record["criterion_text"])},
    ]


def gold_evidence(record):
    return sorted(set(json.loads(record["expert_sentences"])))


def target_text(record):
    return json.dumps({"label": record["expert_eligibility"], "evidence_sentence_ids": gold_evidence(record)})


def parse_output(text, criterion_type, n_sentences):
    """Parse a generation into (label, evidence_ids, ok).

    Labels outside the criterion type's label set become INVALID rather than
    being silently mapped to another class. Evidence IDs outside the note are dropped.
    """
    allowed = LABELS[criterion_type.strip().lower()]
    label, ids, ok = INVALID, [], False
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            cand = str(obj.get("label", "")).strip().lower()
            if cand in allowed:
                label, ok = cand, True
            raw = obj.get("evidence_sentence_ids", [])
            ids = sorted({int(i) for i in raw if str(i).lstrip("-").isdigit() and 0 <= int(i) < n_sentences})
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            pass
    if not ok:
        # Fallback: an allowed label quoted verbatim in malformed output (longest first so
        # "not included" is not read as "included").
        for cand in sorted(allowed, key=len, reverse=True):
            if re.search(rf'"{re.escape(cand)}"', text.lower()):
                label = cand
                break
    return label, ids, ok
