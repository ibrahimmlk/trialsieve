"""Manual Qwen2 forward and decoding loops vs Hugging Face (Qwen2.5-0.5B, fp32, CPU)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trialsieve.decoding import greedy_kv, greedy_no_cache, speculative  # noqa: E402
from trialsieve.manual_model import ManualQwen2  # noqa: E402

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = ["Hi", "List three symptoms of myocardial infarction.",
           "Patient note: 0. A 58-year-old woman presents with chest pain. 1. She has hypertension and obesity. "
           "2. She denies smoking or diabetes. Criterion: Patients aged 18 to 65 with stable angina. Answer in JSON."]


@pytest.fixture(scope="module")
def models():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NAME)
    hf = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    return tok, hf, ManualQwen2(hf)


def test_forward_matches_hf(models):
    tok, hf, m = models
    ids = tok(PROMPTS[2], return_tensors="pt").input_ids
    with torch.no_grad():
        ref = hf(ids).logits
    for att in ["reference", "tiled", "sdpa"]:
        m.set_attention(att)
        torch.testing.assert_close(m.forward(ids), ref, atol=2e-4, rtol=1e-4)
        cache = m.new_cache(ids.shape[1])
        chunks = [m.forward(ids[:, a:b], cache) for a, b in [(0, 7), (7, 8), (8, ids.shape[1])]]
        torch.testing.assert_close(torch.cat(chunks, 1), ref, atol=2e-4, rtol=1e-4)
    m.set_attention("reference")


@pytest.mark.parametrize("prompt,n_new", [(PROMPTS[0], 8), (PROMPTS[1], 24), (PROMPTS[2], 40)])
def test_kv_cache_matches_no_cache_and_hf(models, prompt, n_new):
    tok, hf, m = models
    ids = tok(prompt).input_ids
    no_cache, _ = greedy_no_cache(m, ids, n_new)
    cached, stats = greedy_kv(m, ids, n_new)
    with torch.no_grad():
        hf_out = hf.generate(torch.tensor([ids]), do_sample=False, max_new_tokens=n_new, min_new_tokens=n_new, repetition_penalty=1.0,
                             pad_token_id=tok.eos_token_id)[0, len(ids):].tolist()
    assert cached == no_cache == hf_out
    assert stats["cache_len"] == len(ids) + n_new - 1  # last generated token is never fed back


@pytest.mark.parametrize("k", [1, 3, 5])
def test_speculative_equals_target_greedy(models, k):
    """Draft = the same weights with small noise, so some drafts are rejected and rollback is exercised."""
    from transformers import AutoModelForCausalLM
    tok, _, target = models
    noisy = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in noisy.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.02 * p.std())
    draft = ManualQwen2(noisy)
    ids = tok(PROMPTS[2]).input_ids
    ref, _ = greedy_kv(target, ids, 40)
    out, stats = speculative(target, draft, ids, 40, k=k)
    assert out == ref
    assert stats["accepted"] < stats["drafted"], "test should exercise rejections"


def test_prefix_reuse_matches_fresh_cache(models):
    """Several criteria for one patient: reusing the shared prefix must not change any output."""
    tok, _, m = models
    base = PROMPTS[2].split("Criterion:")[0]
    prompts = [tok(base + c).input_ids for c in
               ["Criterion: Age over 18.", "Criterion: History of diabetes.", "Criterion: Current smoker."]]
    state = {}
    for ids in prompts:
        fresh, _ = greedy_kv(m, ids, 12)
        reused, stats = greedy_kv(m, ids, 12, prefix_state=state)
        assert reused == fresh
    assert stats["reused_prefix_tokens"] > 0


@pytest.mark.parametrize("attention", ["reference", "tiled", "sdpa"])
def test_batched_ragged_decoding_matches_single(models, attention):
    """Criteria of different lengths decoded together must equal decoding each one alone."""
    from trialsieve.decoding import greedy_kv_batch
    tok, _, m = models
    m.set_attention(attention)
    base = PROMPTS[2].split("Criterion:")[0]
    prompts = [tok(base + c).input_ids for c in
               ["Criterion: Age over 18.", "Criterion: History of diabetes mellitus type 2 requiring insulin therapy.",
                "Criterion: Smoker.", "Criterion: Pregnant or breastfeeding women, or women planning pregnancy."]]
    eos = {tok.eos_token_id}
    singles = [greedy_kv(m, ids, 20, eos_ids=eos)[0] for ids in prompts]
    batched, stats = greedy_kv_batch(m, prompts, 20, eos_ids=eos)
    m.set_attention("reference")
    assert batched == singles
    assert stats["shared_prefix_tokens"] > 0
