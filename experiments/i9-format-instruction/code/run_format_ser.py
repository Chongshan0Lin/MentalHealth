#!/usr/bin/env python3
"""I9 -- is a bare FORMAT instruction a shortcut mitigation, or another
flattening?

Found accidentally while adapting the RL design: adding any instruction about
output format -- not an anti-shortcut instruction, just "put your answer in
<answer></answer>" -- collapses the counterfactual effect by 72-119 % depending
on placement. That is a larger reduction than our explicit anti-shortcut
instruction (P1, -53 %) achieves.

The shape is also the shape of chain-of-thought's failure: best-in-study
shortcut number, worst-in-study model. So this cannot be reported as a
mitigation until it has been through the same criteria as everything else:

  CSS              shortcut sensitivity, cf_depression vs cf_neutral   lower better
  construct sens   the frozen 200 E5 pairs, interviewer fixed          higher better
  SER              CSS / construct sens                                lower better
  IID AUC          utility guard on cf_none                            flat
  P(pred=Yes)      degeneracy check                                    not extreme

Everything is measured at the A2b truncation budget with the original template
as the reference condition, so the numbers land on the same scale as the rest of
the paper (baseline: CSS 0.2057, construct 0.1154, SER 1.78, AUC 0.777).
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
E5 = f"{ROOT}/experiments/suite/results/e5_construct_pairs.jsonl"
OUT = f"{ROOT}/experiments/i9-format-instruction/results"
LABELS = ["Yes", "No"]
sig = lambda x: 1 / (1 + np.exp(-x))

_spec = importlib.util.spec_from_file_location(
    "run_a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
_a2b = importlib.util.module_from_spec(_spec)
sys.modules["run_a2b"] = _a2b
_spec.loader.exec_module(_a2b)
POOLS, interleave = dict(_a2b.POOLS), _a2b.interleave
POOLS["cf_none"] = []

BASE = ('Consider this post: "{text}" '
        'Question: Does the poster suffer from depression?')
CONDITIONS = {
    "base": BASE,
    "fmt_prefix": ("Reason inside <think></think>, then give exactly one word, "
                   "Yes or No, inside <answer></answer>.\n\n" + BASE),
    "fmt_suffix": (BASE + " Reason in <think></think>, answer in "
                   "<answer></answer>."),
}


def auc(y, s):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=4032)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    sc = TextScorer(f"{ROOT}/checkpoints/{args.model}", device=args.device)
    rows = [json.loads(l) for l in open(COND)]
    pairs = [json.loads(l) for l in open(E5)]
    y = np.array([r["phq8_binary"] for r in rows])

    # A2b truncation policy: C truncated once, reused across arms, so no arm
    # differs in how much participant text the model saw
    longest = max(len(" ".join(p)) for p in POOLS.values())
    budget = args.budget - (longest // 3) - 64
    C = []
    for r in rows:
        ids = sc.tok(r["text_participant_only"], add_special_tokens=False).input_ids
        C.append(sc.tok.decode(ids[:budget]) if len(ids) > budget
                 else r["text_participant_only"])

    def s1(tmpl, text):
        lp = sc.score(tmpl.format(text=text), LABELS)
        return lp["Yes"]["mean"] - lp["No"]["mean"]

    res = {}
    for name, tmpl in CONDITIONS.items():
        S = {a: np.array([s1(tmpl, interleave(c, POOLS[a])) for c in C])
             for a in ["cf_none", "cf_neutral", "cf_depression"]}
        pos = np.array([s1(tmpl, " ".join(p["pos"]["text"].split())) for p in pairs])
        neg = np.array([s1(tmpl, " ".join(p["neg"]["text"].split())) for p in pairs])

        css = float(np.abs(sig(S["cf_depression"]) - sig(S["cf_neutral"])).mean())
        cs = float(np.abs(sig(pos) - sig(neg)).mean())
        conf = np.where(y == 0, S["cf_depression"], S["cf_neutral"])
        res[name] = dict(
            css=css, construct=cs, ser=css / max(cs, 1e-9),
            iid_auc=auc(y, S["cf_none"]), conflict_auc=auc(y, conf),
            p_yes=float((S["cf_none"] > 0).mean()),
            pair_acc=float((pos > neg).mean()),
            flips=float(np.mean((S["cf_depression"] > 0) != (S["cf_neutral"] > 0))))
        print(f"  {name} done", flush=True)

    print("\n" + "=" * 96)
    print("I9 -- format instruction, evaluated on the full criteria")
    print("=" * 96)
    print(f"{'condition':12s} {'CSS':>7s} {'vs base':>8s} {'construct':>10s} "
          f"{'vs base':>8s} {'SER':>6s} {'IID AUC':>8s} {'conflict':>9s} "
          f"{'P(Yes)':>7s} {'flips':>6s}")
    print("-" * 96)
    b = res["base"]
    for n, r in res.items():
        print(f"{n:12s} {r['css']:7.4f} {r['css']/b['css']:7.2f}x "
              f"{r['construct']:10.4f} {r['construct']/b['construct']:7.2f}x "
              f"{r['ser']:6.2f} {r['iid_auc']:8.3f} {r['conflict_auc']:9.3f} "
              f"{r['p_yes']:7.3f} {r['flips']:6.3f}")

    print("\n" + "=" * 96); print("VERDICT"); print("=" * 96)
    for n, r in res.items():
        if n == "base":
            continue
        dcss = 1 - r["css"] / b["css"]
        dcon = 1 - r["construct"] / b["construct"]
        dauc = r["iid_auc"] - b["iid_auc"]
        ok_u = dauc > -0.05
        ok_c = dcon < dcss - 0.10
        degen = r["p_yes"] > 0.9 or r["p_yes"] < 0.1
        print(f"\n  {n}: shortcut {-dcss:+.1%}, construct {-dcon:+.1%}, "
              f"AUC {dauc:+.3f}, P(Yes) {r['p_yes']:.3f}")
        if degen:
            print("    FAIL -- degenerate operating point.")
        elif ok_u and ok_c:
            print("    PASS -- shortcut falls faster than construct AND utility")
            print("            holds. A genuine inference-level mitigation.")
        elif not ok_u:
            print("    FAIL -- utility lost; this is flattening, like CoT.")
        else:
            print("    FAIL -- construct sensitivity falls as fast as the "
                  "shortcut.")
    json.dump(res, open(f"{OUT}/format_ser.json", "w"), indent=2)
    print(f"\nwrote {OUT}/format_ser.json")


if __name__ == "__main__":
    main()
