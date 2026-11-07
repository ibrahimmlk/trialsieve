"""Greedy decoding loops on top of ManualQwen2: uncached reference, KV-cached, and speculative.

All functions take a 1-D prompt token list and return (new_tokens, stats). Batch size is 1:
the application screens one criterion at a time, and B=1 keeps cache rollback exact.
"""
import time

import torch


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _start_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated()
    return 0


def _peak_memory(device, base):
    return (torch.cuda.max_memory_allocated() - base) if device.type == "cuda" else None


def _finish(tokens, stats, prefill_s, decode_s, device, mem0):
    n = len(tokens)
    stats.update({
        "new_tokens": n, "prefill_s": prefill_s, "decode_s": decode_s, "total_s": prefill_s + decode_s,
        # Decode throughput excludes the first token, which is produced by the prefill pass.
        "decode_tok_per_s": (n - 1) / decode_s if n > 1 and decode_s > 0 else None,
        "e2e_tok_per_s": n / (prefill_s + decode_s) if n else None,
        "peak_mem_bytes": _peak_memory(device, mem0),
    })
    return tokens, stats


def greedy_no_cache(model, prompt, max_new_tokens, eos_ids=(), **_):
    """Reference path: every step re-runs the whole sequence; no state is reused."""
    dev = model.device
    mem0 = _start_memory(dev)
    seq = torch.tensor([prompt], device=dev)
    out, prefill_s = [], 0.0
    _sync(dev); t0 = time.perf_counter()
    for step in range(max_new_tokens):
        nxt = int(model.forward(seq, cache=None, last_only=True)[0, -1].argmax())
        if step == 0:
            _sync(dev); prefill_s = time.perf_counter() - t0
        out.append(nxt)
        if nxt in eos_ids:
            break
        seq = torch.cat([seq, torch.tensor([[nxt]], device=dev)], dim=1)
    _sync(dev)
    return _finish(out, {"mode": "no_cache"}, prefill_s, time.perf_counter() - t0 - prefill_s, dev, mem0)


def greedy_kv(model, prompt, max_new_tokens, eos_ids=(), prefix_state=None, **_):
    """Prefill the prompt once into the KV cache, then feed one token per step at position cache.length.

    prefix_state (optional dict) enables prefix reuse across calls: it keeps the previous
    cache and the tokens it holds. The cache is rolled back to the longest common token
    prefix with the new prompt (e.g. system prompt + patient note shared by every criterion),
    and only the remaining prompt tokens are prefilled.
    """
    dev = model.device
    mem0 = _start_memory(dev)
    need = len(prompt) + max_new_tokens
    reused = 0
    cache = prefix_state.get("cache") if prefix_state is not None else None
    if cache is not None and cache.max_len >= need:
        cached = prefix_state["tokens"][: cache.length]
        while reused < min(len(cached), len(prompt) - 1) and cached[reused] == prompt[reused]:
            reused += 1  # keep >= 1 prompt token to feed, so the last position yields logits
        cache.rollback(reused)
    else:
        # Without reuse allocate exactly what is needed (keeps benchmark memory honest).
        cache = model.new_cache(max(need, 2048) if prefix_state is not None else need)
    _sync(dev); t0 = time.perf_counter()
    nxt = int(model.forward(torch.tensor([prompt[reused:]], device=dev), cache, last_only=True)[0, -1].argmax())
    _sync(dev); prefill_s = time.perf_counter() - t0
    out = [nxt]
    while len(out) < max_new_tokens and nxt not in eos_ids:
        nxt = int(model.forward(torch.tensor([[nxt]], device=dev), cache, last_only=True)[0, -1].argmax())
        out.append(nxt)
    _sync(dev)
    if prefix_state is not None:
        prefix_state["cache"], prefix_state["tokens"] = cache, list(prompt) + out[:-1]
    stats = {"mode": "kv_cache", "cache_len": cache.length, "cache_bytes": cache.nbytes(), "reused_prefix_tokens": reused}
    return _finish(out, stats, prefill_s, time.perf_counter() - t0 - prefill_s, dev, mem0)


def speculative(target, draft, prompt, max_new_tokens, eos_ids=(), k=4, **_):
    """Greedy speculative decoding (Leviathan et al., 2023; Chen et al., 2023), temperature 0.

    Invariant: `seq` holds prompt + committed tokens; the target cache holds seq[:-1], i.e.
    everything except the newest committed token. Each round:
      1. draft catches up on uncached committed tokens, then proposes d1..dk greedily;
      2. target scores [seq[-1], d1..dk] in ONE forward pass, giving its own greedy choice
         t1..t(k+1) after each prefix;
      3. accept the longest prefix with d_i == t_i (n tokens), then append t(n+1), the
         target's correction (or a free bonus token if all k matched);
      4. roll both caches back so they contain no rejected draft tokens.
    The committed tokens are exactly what greedy decoding of the target would produce.
    """
    dev = target.device
    mem0 = _start_memory(dev)
    max_len = len(prompt) + max_new_tokens + k + 1
    t_cache, d_cache = target.new_cache(max_len), draft.new_cache(max_len)
    stats = {"mode": "speculative", "k": k, "rounds": 0, "drafted": 0, "accepted": 0,
             "target_forwards": 0, "draft_forwards": 0, "accept_hist": [0] * (k + 1)}

    _sync(dev); t0 = time.perf_counter()
    # Prefill: the target processes the prompt and produces the first token itself.
    first = int(target.forward(torch.tensor([prompt], device=dev), t_cache, last_only=True)[0, -1].argmax())
    stats["target_forwards"] += 1
    _sync(dev); prefill_s = time.perf_counter() - t0
    seq = list(prompt) + [first]
    out = [first]

    while len(out) < max_new_tokens and out[-1] not in eos_ids:
        # Target cache must hold seq[:-1]. After prefill it holds the prompt = seq[:-1].
        assert t_cache.length == len(seq) - 1
        # 1. Draft: catch up, then propose k tokens (dk is produced but not fed back).
        n_draft = min(k, max_new_tokens - len(out))
        pending = seq[d_cache.length:]
        logits = draft.forward(torch.tensor([pending], device=dev), d_cache, last_only=True)
        stats["draft_forwards"] += 1
        proposal = [int(logits[0, -1].argmax())]
        while len(proposal) < n_draft and proposal[-1] not in eos_ids:
            logits = draft.forward(torch.tensor([[proposal[-1]]], device=dev), d_cache, last_only=True)
            stats["draft_forwards"] += 1
            proposal.append(int(logits[0, -1].argmax()))

        # 2. Verify all proposals with one target pass over [seq[-1], d1..dk].
        verify = torch.tensor([[seq[-1]] + proposal], device=dev)
        target_choice = target.forward(verify, t_cache)[0].argmax(-1).tolist()  # len(proposal) + 1
        stats["target_forwards"] += 1

        # 3. Longest matching prefix, then the target's own next token.
        n = 0
        while n < len(proposal) and proposal[n] == target_choice[n]:
            n += 1
        new = proposal[:n] + [target_choice[n]]
        stats["rounds"] += 1
        stats["drafted"] += len(proposal)
        stats["accepted"] += n
        stats["accept_hist"][n] += 1

        # Stop at EOS / length limit inside the accepted block.
        for i, tkn in enumerate(new):
            if tkn in eos_ids or len(out) + i + 1 >= max_new_tokens:
                new = new[: i + 1]
                break
        seq += new
        out += new

        # 4. Caches: keep only committed tokens (target: seq[:-1]; draft: at most that).
        t_cache.rollback(len(seq) - 1)
        d_cache.rollback(min(d_cache.length, len(seq) - 1))

    _sync(dev)
    stats["acceptance_rate"] = stats["accepted"] / stats["drafted"] if stats["drafted"] else None
    stats["tokens_per_target_forward"] = len(out) / stats["target_forwards"]
    return _finish(out, stats, prefill_s, time.perf_counter() - t0 - prefill_s, dev, mem0)


DECODERS = {"no_cache": greedy_no_cache, "kv_cache": greedy_kv}
