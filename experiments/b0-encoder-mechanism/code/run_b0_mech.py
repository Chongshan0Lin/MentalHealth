#!/usr/bin/env python3
"""B0-mech -- does a DAIC-trained encoder show the SAME shortcut mechanism as
the zero-shot LLM?

Step 2 of INTERVENTION_DESIGN.md §7. This gates the whole train-time arm.

I1 established that on MentaLLaMA the interviewer shortcut is a lexically
triggered *threshold offset*: re-thresholding erases 97.9 % of the
counterfactual accuracy collapse, and deleting the interviewer turns at
inference buys +0.126 AUC. Nothing was learned, so nothing had to be unlearned.

A MentalBERT fine-tuned on DAIC-107 *does* have a training history. If it
absorbs the interviewer cue into its weights, re-thresholding should NOT repair
its counterfactual collapse -- and PoE / group DRO / adversarial become the
matched tools. If re-thresholding does repair it, the mechanisms are the same
and the train-time arm is unnecessary.

Three measurements, per INTERVENTION_DESIGN.md §3:
  1. train `full`       -> test AUC          : does it learn anything at n=107?
  2. train `ellie_only` -> test AUC          : is the shortcut learnable here?
                                               (TF-IDF reached 0.810)
  3. the `full`-trained model over the A2 counterfactual arms, scored with
     acc@default vs acc@opt                  : THE MECHANISM TEST

Bound to the official 107/34/47 splits. Unlike I1 this arm trains, so the
leave-one-out threshold device is unavailable and the numbers are
correspondingly underpowered -- stated, not hidden.

⚠ Gotcha 18: MentalBERT/MentalRoBERTa ship no pooler weights; `pooler.dense.*`
is randomly initialised at load. Mean-pooling only, never `pooler_output`.
⚠ Transcripts run ~3.8k tokens against BERT's 512. Head truncation would delete
the mid-interview region where the clinical follow-ups live (Burdisso 2024),
i.e. the effect under test. Chunk and pool instead.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata
from transformers import AutoModel, AutoTokenizer

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
OUT = f"{ROOT}/experiments/b0-encoder-mechanism/results"
MAX_CHUNKS = 16          # 16 x 512 = 8192 tokens; covers every transcript
rng = np.random.default_rng(0)


# -- reuse the A2b counterfactual construction verbatim ---------------------
# Importing rather than copying guarantees the encoder sees byte-identical
# interviewer pools and identical insertion positions to the LLM run.
_spec = importlib.util.spec_from_file_location(
    "run_a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
_a2b = importlib.util.module_from_spec(_spec)
sys.modules["run_a2b"] = _a2b
_spec.loader.exec_module(_a2b)
POOLS, interleave = _a2b.POOLS, _a2b.interleave


def auc(y, s):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def best_threshold(s, y):
    c = np.unique(np.concatenate([s, [s.min() - 1, s.max() + 1]]))
    a = np.array([((s > t).astype(int) == y).mean() for t in c])
    b = c[a == a.max()]
    return float(b[np.argmin(np.abs(b - np.median(s)))])


def acc_opt(s, y):
    return float(max(((s > t).astype(int) == y).mean()
                     for t in np.unique(np.concatenate([s, [s.min() - 1]]))))


# ------------------------------------------------------------------ model

class ChunkedEncoder(nn.Module):
    """Document encoder over 512-token chunks, with a switchable readout.

    The readout is the control variable for the confound in ../analysis.md §4.
    Under `mean`, a span's influence on the document vector is proportional to
    its token count, so a short but semantically loaded insertion cannot be
    upweighted. `attn` can upweight it; `cls` is an intermediate point (BERT's
    [CLS] attends within a chunk but chunks are still averaged).

    If CF-Sens is flat under `attn` too, the encoder's insensitivity is a
    property of the model rather than of the readout.
    """

    def __init__(self, ckpt, readout="mean"):
        super().__init__()
        self.enc = AutoModel.from_pretrained(ckpt, add_pooling_layer=False)
        H = self.enc.config.hidden_size
        self.readout = readout
        self.head = nn.Linear(H, 2)
        if readout == "attn":
            self.scorer = nn.Linear(H, 1)

    def embed(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        if self.readout == "mean":
            per_chunk = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return per_chunk.mean(0)
        if self.readout == "cls":
            return h[:, 0].mean(0)
        if self.readout == "attn":
            # attention over every real token in the whole document at once,
            # so one chunk can dominate if its content warrants it
            flat_h = h.reshape(-1, h.size(-1))                   # [C*T, H]
            flat_m = mask.reshape(-1).bool()
            s = self.scorer(flat_h).squeeze(-1)
            s = s.masked_fill(~flat_m, float("-inf"))
            w = torch.softmax(s, dim=0).unsqueeze(-1)
            return (flat_h * w).sum(0)
        raise ValueError(f"unknown readout {self.readout}")

    def forward(self, ids, mask):
        return self.head(self.embed(ids, mask))


def chunk(tok, text, device):
    ids = tok(text, add_special_tokens=False).input_ids
    size = tok.model_max_length - 2
    pieces = [ids[i:i + size] for i in range(0, len(ids), size)][:MAX_CHUNKS]
    if not pieces:
        pieces = [[]]
    # `build_inputs_with_special_tokens` was removed in recent transformers;
    # BERT's framing is [CLS] … [SEP], applied explicitly so the chunk layout
    # does not depend on the library version.
    cls_, sep = tok.cls_token_id, tok.sep_token_id
    built = [[cls_] + p + [sep] for p in pieces]
    width = max(len(b) for b in built)
    I = torch.full((len(built), width), tok.pad_token_id, dtype=torch.long)
    M = torch.zeros((len(built), width), dtype=torch.long)
    for i, b in enumerate(built):
        I[i, :len(b)] = torch.tensor(b); M[i, :len(b)] = 1
    return I.to(device), M.to(device), len(ids)


# ------------------------------------------------------------------ train

def run_seed(ckpt, tok, train, dev, test, field, seed, device,
             epochs=20, lr=2e-5, patience=5, accum=8, readout="mean"):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ChunkedEncoder(ckpt, readout=readout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()

    def score(rows):
        model.eval(); out = []
        with torch.no_grad():
            for r in rows:
                I, M, _ = chunk(tok, r[field], device)
                z = model(I, M)
                out.append(float(z[1] - z[0]))
        return np.array(out)

    y_dev = np.array([r["phq8_binary"] for r in dev])
    best, best_state, bad = -1.0, None, 0
    order = np.arange(len(train))
    for ep in range(epochs):
        model.train(); np.random.shuffle(order); opt.zero_grad()
        for k, i in enumerate(order):
            r = train[i]
            I, M, _ = chunk(tok, r[field], device)
            z = model(I, M).unsqueeze(0)
            loss = lossf(z, torch.tensor([r["phq8_binary"]], device=device))
            (loss / accum).backward()
            if (k + 1) % accum == 0 or k == len(order) - 1:
                opt.step(); opt.zero_grad()
        a = auc(y_dev, score(dev))
        if a > best:
            best, bad = a, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{ROOT}/checkpoints/MentalBERT")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--readout", default="mean", choices=["mean", "cls", "attn"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fields", default="text_full,text_ellie_only",
                    help="training conditions; text_full must come first "
                         "because the M3 mechanism test uses that model")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    rows = [json.loads(l) for l in open(COND)]

    # counterfactual arm texts, built exactly as in A2b but WITHOUT the LLM
    # token budget -- the encoder chunks, so C is never truncated. Recorded as
    # a scope difference rather than silently equalised.
    for r in rows:
        for arm in ["cf_none"] + list(POOLS):
            r[arm] = interleave(r["text_participant_only"], POOLS.get(arm, []))

    S = {s: [r for r in rows if r["split"] == s] for s in ["train", "dev", "test"]}
    y_test = np.array([r["phq8_binary"] for r in S["test"]])
    maj = max(y_test.mean(), 1 - y_test.mean())
    print(f"splits: train {len(S['train'])} dev {len(S['dev'])} test {len(S['test'])}"
          f" | test majority {maj:.3f}", flush=True)

    tag = args.tag or args.readout
    print(f"readout = {args.readout}   tag = {tag}", flush=True)
    results = {"ckpt": args.ckpt, "readout": args.readout, "measurements": []}

    # ---- measurements 1 and 2 -------------------------------------------
    trained = {}
    fields = args.fields.split(",")
    assert fields[0] == "text_full", "M3 needs the full-trained model"
    for k, field in enumerate(fields, start=1):
        label = f"M{k} train={field.replace('text_', '')}"
        aucs, accs, models = [], [], []
        for sd in args.seeds:
            t0 = time.time()
            m, dev_auc, score = run_seed(args.ckpt, tok, S["train"], S["dev"],
                                         S["test"], field, sd, args.device,
                                         readout=args.readout)
            s_test = score(S["test"])
            s_dev = score(S["dev"])
            thr = best_threshold(s_dev, np.array([r["phq8_binary"] for r in S["dev"]]))
            a = auc(y_test, s_test)
            ac = float(((s_test > thr).astype(int) == y_test).mean())
            aucs.append(a); accs.append(ac); models.append((m, score))
            print(f"  {label} seed{sd}: dev_auc {dev_auc:.3f} | "
                  f"test AUC {a:.3f} acc {ac:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
        trained[field] = models
        print(f"{label}: test AUC {np.mean(aucs):.3f} "
              f"(sd {np.std(aucs):.3f}, seeds {[f'{x:.3f}' for x in aucs]})\n",
              flush=True)
        results["measurements"].append(
            dict(name=label, field=field, auc_mean=float(np.mean(aucs)),
                 auc_sd=float(np.std(aucs)), aucs=aucs, accs=accs))

    # ---- measurement 3: THE MECHANISM TEST -------------------------------
    print("M3 MECHANISM TEST -- full-trained encoder over the A2 arms")
    print(f"  {'arm':16s} {'AUC':>6s} {'acc@dev-thr':>12s} {'acc@opt':>8s} {'P(Yes)':>7s}")
    arms = ["cf_neutral", "cf_none", "cf_empirical", "cf_depression"]
    per_arm = {a: {"auc": [], "acc_def": [], "acc_opt": [], "p_yes": []}
               for a in arms}
    cf_sens = []
    y_dev = np.array([r["phq8_binary"] for r in S["dev"]])
    for (m, _score) in trained["text_full"]:
        sc = {}
        for a in arms:
            # the arm text is bound into a fixed field name so the scorer reads
            # the counterfactual transcript rather than `text_full`
            st = _score_field(m, tok, [{**r, "ARM": r[a]} for r in S["test"]],
                              "ARM", args.device)
            sd_ = _score_field(m, tok, [{**r, "ARM": r[a]} for r in S["dev"]],
                               "ARM", args.device)
            thr = best_threshold(sd_, y_dev)
            sc[a] = st
            per_arm[a]["auc"].append(auc(y_test, st))
            per_arm[a]["acc_def"].append(float(((st > thr).astype(int) == y_test).mean()))
            per_arm[a]["acc_opt"].append(acc_opt(st, y_test))
            per_arm[a]["p_yes"].append(float((st > thr).mean()))
        sig = lambda x: 1 / (1 + np.exp(-x))
        cf_sens.append(float(np.abs(sig(sc["cf_depression"]) - sig(sc["cf_neutral"])).mean()))

    for a in arms:
        d = per_arm[a]
        print(f"  {a:16s} {np.mean(d['auc']):6.3f} {np.mean(d['acc_def']):12.3f} "
              f"{np.mean(d['acc_opt']):8.3f} {np.mean(d['p_yes']):7.3f}")
    results["mechanism"] = {a: {k: [float(x) for x in v] for k, v in d.items()}
                            for a, d in per_arm.items()}
    results["cf_sens"] = cf_sens

    dn = np.mean(per_arm["cf_neutral"]["acc_def"]) - np.mean(per_arm["cf_depression"]["acc_def"])
    do = np.mean(per_arm["cf_neutral"]["acc_opt"]) - np.mean(per_arm["cf_depression"]["acc_opt"])
    print(f"\n  neutral - clinical accuracy gap @ dev threshold : {dn:+.3f}")
    print(f"  neutral - clinical accuracy gap @ optimal        : {do:+.3f}")
    frac = 1 - do / dn if abs(dn) > 1e-9 else float("nan")
    print(f"  fraction of the collapse that is threshold shift : {frac:.1%}")
    print(f"  CF-Sens (mean over seeds)                        : {np.mean(cf_sens):.4f}")
    print("\n  LLM comparison (I1): gap 0.256 -> 0.008, 97.9 % threshold shift,"
          " CF-Sens 0.1932")
    results["summary"] = dict(gap_default=float(dn), gap_optimal=float(do),
                              frac_threshold=float(frac),
                              cf_sens_mean=float(np.mean(cf_sens)))

    with open(f"{OUT}/b0_mech_summary_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}/b0_mech_summary_{tag}.json")


def _score_field(model, tok, rows, field, device):
    model.eval(); out = []
    with torch.no_grad():
        for r in rows:
            I, M, _ = chunk(tok, r[field], device)
            z = model(I, M)
            out.append(float(z[1] - z[0]))
    return np.array(out)


if __name__ == "__main__":
    main()
