#!/usr/bin/env python3
"""Label log-likelihood scoring, shared by every experiment.

Predictions are read as a distribution over a closed label space via
teacher-forced log-likelihood, never by parsing free generation: parse-failure
rate correlates with perturbation strength, which would manufacture fake
accuracy drops in exactly the ablations we care about.

Every scorer returns, per label, both `sum` and `mean` token log-prob, plus the
same quantities under a null prompt so PMI normalization
(`log p(y|x) - log p(y|null)`) can be applied downstream.  The gap between raw
and PMI-normalized accuracy is itself the label-prior measurement.
"""
from __future__ import annotations

import numpy as np
import torch

# Qwen2.5-Omni chat EOS/pad. OmniSapiens2.0 and OmniSapiens-RL ship an empty
# generation_config.json, so these must be pinned explicitly (see gotcha 8).
IM_END, ENDOFTEXT = "<|im_end|>", "<|endoftext|>"


class LabelScorer:
    """Base: turns a prompt + label space into per-label log-probabilities."""

    def _forward_logprobs(self, full_ids, prompt_len, **kw):
        """Mean/sum log-prob of the tokens after `prompt_len`."""
        with torch.no_grad():
            logits = self.model(input_ids=full_ids, **kw).logits
        lp = logits[0, :-1].log_softmax(-1)
        tgt = full_ids[0, 1:]
        n = full_ids.shape[1] - prompt_len
        if n <= 0:
            raise ValueError("label produced no tokens")
        tok_lp = lp[-n:].gather(1, tgt[-n:, None]).squeeze(1)
        return float(tok_lp.sum()), float(tok_lp.mean()), n

    def score(self, prompt, label_space, media=None, null_prompt=None):
        """-> {label: {sum, mean, n_tok, null_sum, null_mean}}"""
        out = {}
        scored = self._score_all(prompt, label_space, media)
        for lab, (s, m, n) in zip(label_space, scored):
            out[lab] = {"sum": s, "mean": m, "n_tok": n}
        if null_prompt is not None:
            for lab, (s, m, _) in zip(label_space,
                                      self._score_all(null_prompt, label_space, None)):
                out[lab]["null_sum"], out[lab]["null_mean"] = s, m
        return out

    def _score_all(self, prompt, label_space, media):
        """Default: independent forward per label. Subclasses may override with a
        prefix-cached version — the prompt is identical across candidates, so
        recomputing it once per label is pure waste (and for a 20k-token
        multimodal prompt it dominates the run)."""
        return [self._score_one(prompt, lab, media) for lab in label_space]

    def _label_ids(self, tok, text, label):
        """Token ids for `label` **as it is actually tokenized in context**.

        Tokenizing the label on its own is wrong and silently so: SentencePiece
        and BPE encode a word differently depending on whether a space precedes
        it, so `tok("Yes")` yields the leading-space variant while the same
        characters inside `prompt + "Yes"` may not. Scoring the standalone ids
        against an in-context prefix mixes the two and shifts log-probs by
        several nats -- enough to flip a prediction.

        Deriving the ids as the suffix of the tokenized concatenation makes the
        cached and uncached paths identical by construction.
        """
        a = tok(text, add_special_tokens=False).input_ids
        b = tok(text + label, add_special_tokens=False).input_ids
        if b[:len(a)] != a:
            # Boundary re-merge: the concatenation retokenizes the tail of the
            # prompt. Fall back to the longest shared prefix so the scored span
            # still covers exactly the differing tail.
            n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
            return b[n:], n
        return b[len(a):], len(a)


class TextScorer(LabelScorer):
    """Causal LMs: MentaLLaMA, Menta, OSim-8B."""

    def __init__(self, ckpt, device="cuda:0", adapter=None, chat=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.chat = chat
        self.tok = AutoTokenizer.from_pretrained(adapter or ckpt)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt, dtype=torch.bfloat16, device_map=device)
        if adapter:                       # Menta ships a LoRA adapter only
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def _render(self, prompt):
        if not self.chat or self.tok.chat_template is None:
            return prompt
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False)

    def _score_one(self, prompt, label, media=None):
        rendered = self._render(prompt)
        p_ids = self.tok(rendered, return_tensors="pt").input_ids
        f_ids = self.tok(rendered + " " + label, return_tensors="pt").input_ids
        return self._forward_logprobs(f_ids.to(self.model.device),
                                      p_ids.shape[1])


class OmniScorer(LabelScorer):
    """Qwen2.5-Omni thinkers: OmniSapiens2.0 / -SFT / -RL, HumanOmniV2.

    media = {"audio": np.ndarray @16kHz | None, "video": bytes | None}
    The prompt must already contain the matching <audio>/<video> placeholders.
    """

    SR = 16000

    def __init__(self, ckpt, device="cuda:0"):
        from transformers import (Qwen2_5OmniProcessor,
                                  Qwen2_5OmniThinkerForConditionalGeneration)
        self.proc = Qwen2_5OmniProcessor.from_pretrained(ckpt)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            ckpt, dtype=torch.bfloat16, device_map=device).eval()
        self.eos = self.proc.tokenizer.convert_tokens_to_ids(IM_END)
        self.pad = self.proc.tokenizer.convert_tokens_to_ids(ENDOFTEXT)

    def _build(self, prompt, media):
        content = []
        media = media or {}
        if media.get("video") is not None:
            content.append({"type": "video", "video": "<placeholder>"})
        if media.get("audio") is not None:
            content.append({"type": "audio", "audio": "<placeholder>"})
        content.append({"type": "text", "text": prompt})
        conv = [{"role": "user", "content": content}]
        return self.proc.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False)

    def _pack(self, text, media):
        kw = {}
        media = media or {}
        if media.get("audio") is not None:
            kw["audio"] = [np.asarray(media["audio"], dtype=np.float32)]
            kw["sampling_rate"] = self.SR
        if media.get("video") is not None:
            kw["videos"] = [media["video"]]
        # input_features stay float32: casting Whisper features to bfloat16
        # silently corrupts them (see gotcha 7).
        return self.proc(text=[text], return_tensors="pt",
                         padding=True, **kw).to(self.model.device)

    def _score_one(self, prompt, label, media=None):
        """Reference implementation: one full forward per label. Kept as the
        correctness oracle for `_score_all`; use `score()` in real runs."""
        text = self._build(prompt, media)
        base = self._pack(text, media)
        p_len = base["input_ids"].shape[1]
        full = self._pack(text + label, media)
        n_lab = full["input_ids"].shape[1] - p_len
        kw = {k: v for k, v in full.items() if k != "input_ids"}
        with torch.no_grad():
            logits = self.model(input_ids=full["input_ids"], **kw).logits
        # Slice to the label window BEFORE the softmax. Normalizing the whole
        # sequence first materializes a (seq_len x vocab) fp32 tensor -- 17 GB
        # for a 20k-token native-resolution prompt against a 152k vocab, which
        # OOMs a 48 GB card for scores we then throw away.
        window = logits[0, -n_lab - 1:-1].float().log_softmax(-1)
        tgt = full["input_ids"][0, -n_lab:]
        t = window.gather(1, tgt[:, None]).squeeze(1)
        return float(t.sum()), float(t.mean()), n_lab

    # Prefix-cached scoring is ~7x faster but NOT equivalent on this
    # architecture, so it is off by default. Qwen2.5-Omni uses M-RoPE: token
    # positions are derived from the audio/video grid, and a continuation
    # forwarded against a cache has no way to recover them, so the label tokens
    # are scored at wrong positions. Measured drift is ~11 nat on MELD items --
    # argmax happened to survive on the rows tried, but the distribution does
    # not, which breaks CF-Sensitivity and every probability-based metric.
    # (The same trick IS exact on 1-D RoPE text models -- verified to 0.00 nat
    # on MentaLLaMA -- so TextScorer could adopt it safely.)
    # To re-enable, feed explicit 3-D position_ids built from the thinker's
    # `rope_deltas` and re-run the equivalence check below.
    USE_PREFIX_CACHE = False

    def _score_all(self, prompt, label_space, media):
        if not self.USE_PREFIX_CACHE:
            return [self._score_one(prompt, lab, media) for lab in label_space]
        from transformers import DynamicCache

        text = self._build(prompt, media)
        inputs = self._pack(text, media)
        p_len = inputs["input_ids"].shape[1]

        cache = DynamicCache()
        with torch.no_grad():
            pre = self.model(**inputs, past_key_values=cache, use_cache=True)
        # Distribution over the *first* label token, produced by the prefix's
        # final position. Keep it around: each label needs it, and it is the one
        # piece the per-label window cannot supply for itself.
        last_prefix_logits = pre.logits[0, -1:].log_softmax(-1)

        tok = self.proc.tokenizer
        out = []
        for lab in label_space:
            lab_ids, keep = self._label_ids(tok, text, lab)
            if keep != len(tok(text, add_special_tokens=False).input_ids):
                raise ValueError(
                    f"label {lab!r} retokenizes the prompt tail; cached scoring "
                    "cannot rewind past the cached prefix")
            ids = torch.tensor([lab_ids], device=self.model.device)
            with torch.no_grad():
                logits = self.model(input_ids=ids, past_key_values=cache,
                                    use_cache=True).logits
            # logits[:, i] predicts token i+1, so the first label token is
            # predicted by the prefix's final position, not by this window.
            lp = logits[0].log_softmax(-1)
            joint = (torch.cat([last_prefix_logits, lp[:-1]], dim=0)
                     if ids.shape[1] > 1 else last_prefix_logits)
            tokl = joint.gather(1, ids[0][:, None]).squeeze(1)
            out.append((float(tokl.sum()), float(tokl.mean()), int(ids.shape[1])))
            cache.crop(p_len)          # rewind so the next label reuses the prefix
        return out


# ---------------------------------------------------------------- utilities

def to_prob(scored, key="mean", pmi=False):
    """Per-label scores -> normalized distribution over the label space."""
    labs = list(scored)
    v = np.array([scored[l][key] for l in labs], dtype=np.float64)
    if pmi:
        nk = "null_" + key
        if nk not in scored[labs[0]]:
            raise ValueError("null_prompt was not scored; cannot apply PMI")
        v = v - np.array([scored[l][nk] for l in labs])
    v = np.exp(v - v.max())
    return dict(zip(labs, v / v.sum()))


def predict(scored, key="mean", pmi=False):
    p = to_prob(scored, key, pmi)
    return max(p, key=p.get), p
