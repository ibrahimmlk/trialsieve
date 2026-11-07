"""Screening engine used by the browser application.

Loads the fine-tuned target (Qwen2.5-1.5B + LoRA, adapter kept unmerged so it can be
switched off for a live before/after comparison) and, on demand, the 0.5B draft model.
All decoding goes through the manual implementation in manual_model.py / decoding.py,
except the optional "HF generate" library reference mode.
"""
import contextlib
import hashlib
import html
import json
import re
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import LABELS, ROOT, load_split, messages, note_sentences, number_note, parse_output
from .decoding import greedy_kv, greedy_kv_batch, greedy_no_cache, speculative
from .manual_model import ManualQwen2

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "ibrahimmlk/trialsieve-qwen2.5-1.5b-lora"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
DRAFT_ADAPTER = "ibrahimmlk/trialsieve-qwen2.5-0.5b-draft-lora"

MODES = {
    "KV cache, all criteria batched (fastest)": "kv_batched",
    "KV cache, one criterion at a time": "kv_cache",
    "No cache (manual reference)": "no_cache",
    "Speculative (manual, 0.5B draft)": "speculative",
    "HF generate (library reference)": "hf_generate",
}
MAX_NOTE_TOKENS = 1200
MAX_CRITERIA = 40
MAX_CRITERION_CHARS = 1000
MAX_NEW_TOKENS = 64

DISCLAIMER = ("Educational screening prototype for a university assignment. It is NOT a clinical enrollment "
              "decision tool; every result must be reviewed by qualified clinical research staff.")

LABEL_MEANING = {
    "included": "the note indicates the patient meets this inclusion criterion",
    "not included": "the note indicates the patient does not meet this inclusion criterion",
    "excluded": "the note indicates the patient meets this exclusion criterion",
    "not excluded": "the note indicates the patient does not meet this exclusion criterion",
    "not enough information": "the note does not contain enough information to decide",
    "not applicable": "the criterion does not apply to this patient",
    "invalid output": "the model did not return a valid label, so no decision is made",
}


class InputError(ValueError):
    """User-facing validation error."""


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def parse_criteria_block(text):
    """One criterion per line; strips bullets/numbering and blank lines."""
    out = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line:
            out.append(line)
    return out


def fetch_trial_criteria(nct_id):
    """Public ClinicalTrials.gov v2 API: returns (title, inclusion list, exclusion list). Needs internet."""
    nct_id = (nct_id or "").strip().upper()
    if not re.fullmatch(r"NCT\d{8}", nct_id):
        raise InputError("Trial ID must look like NCT01234567.")
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}?fields=BriefTitle,EligibilityCriteria"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.load(r)
    except Exception as e:
        raise InputError(f"Could not fetch {nct_id} from ClinicalTrials.gov ({e}). Enter criteria manually.")
    proto = data.get("protocolSection", {})
    title = proto.get("identificationModule", {}).get("briefTitle", nct_id)
    text = proto.get("eligibilityModule", {}).get("eligibilityCriteria", "")
    parts = re.split(r"(?i)exclusion criteria\s*:?", text, maxsplit=1)
    inc = re.sub(r"(?i)inclusion criteria\s*:?", "", parts[0])
    exc = parts[1] if len(parts) > 1 else ""
    return title, parse_criteria_block(inc), parse_criteria_block(exc)


def validation_examples():
    """(patient, trial) groups from the VALIDATION split, for demo inputs. The test split is never used here."""
    groups = {}
    for r in load_split("val"):
        g = groups.setdefault((r["patient_id"], r["trial_id"]), {"note": r["note"], "title": r["trial_title"],
                                                                 "inclusion": [], "exclusion": [], "gold": {}})
        g[r["criterion_type"]].append(r["criterion_text"] or "")
        g["gold"][(r["criterion_type"], (r["criterion_text"] or "").strip())] = r["expert_eligibility"]
    return {f"{p} | {t} ({len(g['inclusion'])} inc, {len(g['exclusion'])} exc)": g for (p, t), g in groups.items()}


def explanation(label, evidence, criterion_type):
    meaning = LABEL_MEANING.get(label, label)
    if evidence:
        cited = "; ".join(f'sentence {i}: "{s}"' for i, s in evidence)
        return f"Predicted '{label}': {meaning}. Cited evidence from the note — {cited}."
    if label == "not enough information":
        return (f"Predicted '{label}': no sentence in the note addresses this {criterion_type} criterion. "
                "Missing information is not treated as evidence for or against eligibility.")
    return f"Predicted '{label}': {meaning}. The model cited no supporting sentence, so this result is unsupported and needs review."


def aggregate(results):
    """Explicit rule-based patient-trial outcome (the dataset has no trial-level ground truth)."""
    failing = [r for r in results if (r["type"], r["label"]) in {("inclusion", "not included"), ("exclusion", "excluded")}]
    unknown = [r for r in results if r["label"] in ("not enough information", "invalid output")]
    set_aside = [r for r in results if r["label"] == "not applicable"]
    # Ignoring N/A criteria is only safe when few are set aside: a model that labels most criteria
    # N/A would otherwise produce "potentially eligible" without having checked anything.
    mostly_na = len(set_aside) > len(results) / 2
    if failing:
        verdict = "Likely ineligible"
        reason = f"{len(failing)} criterion/criteria failed (inclusion not met or exclusion met)."
    elif unknown:
        verdict = "Needs review (insufficient information)"
        reason = f"No criterion failed, but {len(unknown)} could not be decided from the note."
    elif mostly_na:
        verdict = "Needs review (most criteria judged not applicable)"
        reason = "No criterion failed, but most criteria were set aside as not applicable, so too few were checked."
    else:
        verdict = "Potentially eligible"
        reason = "Every applicable inclusion criterion is met and no exclusion criterion is met."
    warning = None
    if set_aside:
        checked = len(results) - len(set_aside)
        warning = (f"{len(set_aside)} of {len(results)} criteria were judged not applicable and ignored; "
                   f"the outcome rests on the remaining {checked}.")
        if mostly_na:
            warning += " Most criteria were set aside, so do not rely on this outcome without manual review."
    return {"verdict": verdict, "reason": reason, "warning": warning,
            "rule": "Ineligible if any inclusion = 'not included' or any exclusion = 'excluded'; otherwise "
                    "'needs review' if any criterion is 'not enough information' or invalid, or if more than half "
                    "of the criteria are 'not applicable'; otherwise potentially eligible. 'not applicable' "
                    "criteria are otherwise ignored.",
            "failing": [r["index"] for r in failing], "missing_information": [r["criterion"] for r in unknown],
            "not_applicable": [r["index"] for r in set_aside]}


class Engine:
    def __init__(self, target=TARGET, adapter=ADAPTER, draft=DRAFT, draft_adapter=DRAFT_ADAPTER, log=print):
        self.names = {"target": target, "adapter": adapter, "draft": draft, "draft_adapter": None}
        self.lock = threading.Lock()
        self.log = log
        self.tok = AutoTokenizer.from_pretrained(target)
        self.eos = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id}
        t0 = time.time()
        base = AutoModelForCausalLM.from_pretrained(target, dtype=torch.float32).eval()
        from peft import PeftModel
        self.peft = PeftModel.from_pretrained(base, adapter).eval()
        # LoRA is merged into the weights for speed; switching to the base model unmerges it.
        self.peft.merge_adapter()
        self.merged = True
        self.target = ManualQwen2(self.peft.base_model.model)
        self.load_seconds = time.time() - t0
        self._draft_adapter_id = draft_adapter
        self.draft = None
        self.prefix_states = {}  # per adapter on/off: reusable KV cache of the last prompt

    def ensure_draft(self):
        if self.draft is None:
            model = AutoModelForCausalLM.from_pretrained(self.names["draft"], dtype=torch.float32).eval()
            if self._draft_adapter_id:
                try:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, self._draft_adapter_id).merge_and_unload().eval()
                    self.names["draft_adapter"] = self._draft_adapter_id
                except Exception as e:  # draft adapter not published yet: fall back to the base draft
                    self.log(f"Draft adapter unavailable ({e}); using base {self.names['draft']}")
            self.draft = ManualQwen2(model)
        return self.draft

    @contextlib.contextmanager
    def adapter(self, enabled):
        """Fine-tuned (LoRA merged) or base model (LoRA unmerged and disabled), same weights in memory."""
        if enabled:
            if not self.merged:
                self.peft.merge_adapter()
                self.merged = True
            yield
        else:
            if self.merged:
                self.peft.unmerge_adapter()
                self.merged = False
            with self.peft.disable_adapter():
                yield

    def prompt_ids(self, note, ctype, text):
        rec = {"note": note, "criterion_type": ctype, "criterion_text": text}
        chat = self.tok.apply_chat_template(messages(rec), tokenize=False, add_generation_prompt=True)
        return self.tok(chat, add_special_tokens=False).input_ids

    def decode(self, ids, mode, attention="reference", k=4, use_adapter=True, reuse_prefix=True,
               max_new_tokens=MAX_NEW_TOKENS):
        self.target.set_attention(attention)
        with self.adapter(use_adapter):
            if mode == "kv_cache":
                state = self.prefix_states.setdefault(use_adapter, {}) if reuse_prefix else None
                return greedy_kv(self.target, ids, max_new_tokens, eos_ids=self.eos, prefix_state=state)
            if mode == "no_cache":
                return greedy_no_cache(self.target, ids, max_new_tokens, eos_ids=self.eos)
            if mode == "speculative":
                return speculative(self.target, self.ensure_draft(), ids, max_new_tokens, eos_ids=self.eos, k=k)
            if mode == "hf_generate":
                t0 = time.perf_counter()
                with torch.no_grad():
                    gen = self.peft.generate(input_ids=torch.tensor([ids]), do_sample=False, repetition_penalty=1.0,
                                             max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
                out = gen[0, len(ids):].tolist()
                dt = time.perf_counter() - t0
                return out, {"mode": "hf_generate", "new_tokens": len(out), "total_s": dt,
                             "e2e_tok_per_s": len(out) / dt if dt else None}
        raise ValueError(mode)

    def validate(self, note, criteria):
        if not note or len(note.strip()) < 20:
            raise InputError("Please enter a patient note (at least a sentence or two).")
        numbered = number_note(note)
        n_tok = len(self.tok(numbered).input_ids)
        if n_tok > MAX_NOTE_TOKENS:
            raise InputError(f"The patient note is {n_tok} tokens; the limit is {MAX_NOTE_TOKENS} "
                             "(the model was trained on notes up to ~300 tokens). Please shorten it.")
        if not criteria:
            raise InputError("Please enter at least one inclusion or exclusion criterion.")
        if len(criteria) > MAX_CRITERIA:
            raise InputError(f"{len(criteria)} criteria given; the limit is {MAX_CRITERIA} per run.")
        for ctype, text in criteria:
            if ctype not in LABELS:
                raise InputError(f"Unknown criterion type '{ctype}'.")
            if len(text) > MAX_CRITERION_CHARS:
                raise InputError(f"A criterion is longer than {MAX_CRITERION_CHARS} characters: '{text[:60]}...'")
        return numbered, n_tok

    def screen(self, note, criteria, mode="kv_cache", attention="reference", k=4, use_adapter=True,
               reuse_prefix=True, progress=None):
        with self.lock:
            numbered, n_tok = self.validate(note, criteria)
            sentences = dict(note_sentences(numbered))
            results, t0 = [], time.time()
            all_ids = [self.prompt_ids(numbered, ctype, text) for ctype, text in criteria]
            if mode == "kv_batched":
                if progress:
                    progress(0.1, desc=f"Decoding {len(criteria)} criteria as one batch")
                self.target.set_attention(attention)
                with self.adapter(use_adapter):
                    outs, bstats = greedy_kv_batch(self.target, all_ids, MAX_NEW_TOKENS, eos_ids=self.eos)
                per = [(o, {"total_s": bstats["total_s"] / len(criteria), "new_tokens": len(o),
                            "reused_prefix_tokens": bstats["shared_prefix_tokens"]}) for o in outs]
            else:
                per = []
                for i, ids in enumerate(all_ids):
                    if progress:
                        progress(i / len(criteria), desc=f"Criterion {i + 1}/{len(criteria)}")
                    per.append(self.decode(ids, mode, attention, k, use_adapter, reuse_prefix))
            for i, ((ctype, text), (out, stats)) in enumerate(zip(criteria, per)):
                raw = self.tok.decode(out, skip_special_tokens=True)
                label, ev_ids, parsed = parse_output(raw, ctype, len(sentences))
                evidence = [(j, sentences[j]) for j in ev_ids if j in sentences]
                results.append({"index": i + 1, "type": ctype, "criterion": text, "label": label,
                                "evidence_ids": ev_ids, "evidence": evidence, "parsed_json": parsed,
                                "explanation": explanation(label, evidence, ctype), "raw_output": raw,
                                "seconds": stats.get("total_s"), "new_tokens": stats.get("new_tokens"),
                                "reused_prefix_tokens": stats.get("reused_prefix_tokens"),
                                "acceptance_rate": stats.get("acceptance_rate")})
            if mode == "kv_batched":
                for r in results:
                    r["seconds"] = None  # one shared batch; total time is in provenance
            summary = aggregate(results)
            settings = {"decoding": mode, "attention": attention, "speculative_k": k if mode == "speculative" else None,
                        "fine_tuned_adapter": use_adapter,
                        "prefix_cache_reuse": mode == "kv_batched" or (reuse_prefix and mode == "kv_cache"),
                        "max_new_tokens": MAX_NEW_TOKENS, "greedy": True}
            provenance = {
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "app_commit": git_commit(), "target_model": self.names["target"],
                "adapter": self.names["adapter"] if use_adapter else None,
                "draft_model": self.names["draft"] if mode == "speculative" else None,
                "draft_adapter": self.names["draft_adapter"] if mode == "speculative" else None,
                "device": "cpu", "dtype": "float32", "note_tokens": n_tok,
                "input_sha256": hashlib.sha256(json.dumps([numbered, criteria]).encode()).hexdigest()[:16],
                "runtime_s": round(time.time() - t0, 2)}
            return {"note": numbered, "sentences": sentences, "results": results, "summary": summary,
                    "settings": settings, "provenance": provenance, "disclaimer": DISCLAIMER}


def report_html(rep, title="Eligibility screening report"):
    e = html.escape
    rows = "".join(
        f"<tr><td>{r['index']}</td><td>{e(r['type'])}</td><td>{e(r['criterion'])}</td><td><b>{e(r['label'])}</b></td>"
        f"<td>{e(', '.join(map(str, r['evidence_ids'])) or '—')}</td><td>{e(r['explanation'])}</td></tr>"
        for r in rep["results"])
    note = "".join(f"<li value='{i}'>{e(s)}</li>" for i, s in rep["sentences"].items())
    s = rep["summary"]
    missing = "".join(f"<li>{e(m)}</li>" for m in s["missing_information"]) or "<li>None</li>"
    warn = f"<p class='warn'><b>Caution:</b> {e(s['warning'])}</p>" if s.get("warning") else ""
    prov = "".join(f"<tr><th>{e(k)}</th><td>{e(str(v))}</td></tr>" for k, v in {**rep["provenance"], **rep["settings"]}.items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1a1a1a}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}td,th{{border:1px solid #ccc;padding:6px;vertical-align:top;text-align:left;font-size:14px}}
.warn{{background:#fff4e5;border:1px solid #f0a020;padding:10px;border-radius:6px}}ol li{{margin:2px 0}}</style></head><body>
<h1>{e(title)}</h1><p class="warn"><b>Notice:</b> {e(rep['disclaimer'])}</p>
<h2>Preliminary outcome: {e(s['verdict'])}</h2><p>{e(s['reason'])}</p>{warn}<p><i>Aggregation rule:</i> {e(s['rule'])}</p>
<h3>Missing information</h3><ul>{missing}</ul>
<h2>Criterion-level results</h2><table><tr><th>#</th><th>Type</th><th>Criterion</th><th>Label</th><th>Evidence IDs</th><th>Explanation</th></tr>{rows}</table>
<h2>Patient note (sentence IDs)</h2><ol start="0">{note}</ol>
<h2>Provenance and settings</h2><table>{prov}</table></body></html>"""
