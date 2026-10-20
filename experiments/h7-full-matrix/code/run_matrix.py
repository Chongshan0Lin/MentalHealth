#!/usr/bin/env python3
"""H7 -- fill the model x dataset counterfactual matrix.

H4/H5/H6 each ran only the model family native to their corpus: OmniSapiens on
the HBA subsets, MentaLLaMA/Menta on IMHI. That was defensible but left DAIC as
the only dataset with cross-family coverage, so a per-dataset table like the
8-model DAIC sweep could not be produced anywhere else.

This runs every available model on all three counterfactuals with the arms,
pools and truncation budgets held exactly as the original experiments defined
them, so new rows are directly comparable to the ones already reported.

Scorer is dispatched on architecture: LlamaForCausalLM / Qwen3ForCausalLM take
the text scorer, Qwen2_5OmniThinker takes the omni scorer with media=None. All
three datasets are text-only, so nothing else differs between the two paths.
"""
from __future__ import annotations

import argparse, glob, json, os, random, re, sys, time

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
sys.path.insert(0, "/data/cl195/NeurIPS_workshop/experiments/h4-mmpsy-counterfactual/code")
sys.path.insert(0, "/data/cl195/NeurIPS_workshop/experiments/h6-irf-register-cf/code")

ROOT = "/data/cl195/NeurIPS_workshop"

MODELS = {
    "MentaLLaMA":      dict(ckpt=f"{ROOT}/checkpoints/MentaLLaMA-chat-7B"),
    "Menta":           dict(ckpt=f"{ROOT}/checkpoints/Qwen3-4B-base",
                            adapter=f"{ROOT}/checkpoints/Menta-LoRA/Menta"),
    "qwen3_base":      dict(ckpt=f"{ROOT}/checkpoints/Qwen3-4B-base"),
    "OSim-8B":         dict(ckpt=f"{ROOT}/checkpoints/OSim-8B"),
    "OmniSapiens-SFT": dict(ckpt=f"{ROOT}/checkpoints/OmniSapiens-SFT"),
    "OmniSapiens-RL":  dict(ckpt=f"{ROOT}/checkpoints/OmniSapiens-RL"),
    "OmniSapiens2.0":  dict(ckpt=f"{ROOT}/checkpoints/OmniSapiens2.0"),
    "HumanOmniV2":     dict(ckpt=f"{ROOT}/checkpoints/HumanOmniV2"),
}


def make_scorer(name, device):
    cfg = MODELS[name]
    arch = json.load(open(f"{cfg['ckpt']}/config.json"))["architectures"][0]
    if arch.startswith("Qwen2_5Omni"):
        from shortcut_harness.scoring import OmniScorer
        sc = OmniScorer(cfg["ckpt"], device=device)
        return lambda prompt, labels: sc.score(prompt, labels, media=None)
    from shortcut_harness.scoring import TextScorer
    sc = TextScorer(cfg["ckpt"], device=device, adapter=cfg.get("adapter"))
    return lambda prompt, labels: sc.score(prompt, labels)


# ---------------------------------------------------------------- mmpsy (H4)
def task_mmpsy(score, out, seed):
    import pyarrow.parquet as pq
    from run_h4 import seg_plus, seg_minus, TPL_MARK, SURFACE
    rows = []
    for s in sorted(glob.glob(f"{ROOT}/data/raw/HBA-test/test-*.parquet")):
        ds = pq.read_table(s, columns=["dataset"]).to_pydict()["dataset"]
        if "mmpsy_depression" in set(ds):
            df = pq.read_table(s).to_pandas()
            rows += [r for _, r in df[df["dataset"] == "mmpsy_depression"].iterrows()]
    labels = ["depression", "no_depression"]
    surf = [SURFACE[l] for l in labels]; back = {SURFACE[l]: l for l in labels}
    n = 0
    with open(out, "w") as f:
        for i, r in enumerate(rows):
            if TPL_MARK not in r["problem"]:
                continue
            head, tail = r["problem"].split(TPL_MARK, 1)
            segs = head.strip().split()
            arms = {"native": segs, "seg_plus": seg_plus(list(segs)),
                    "seg_minus": seg_minus(list(segs))}
            if any(v is None for v in arms.values()):
                continue
            for arm, sv in arms.items():
                sc = score(" ".join(sv) + "\n" + TPL_MARK + tail, surf)
                lp = {back[k]: v["mean"] for k, v in sc.items()}
                pred = max(lp, key=lp.get)
                f.write(json.dumps({"idx": i, "arm": arm, "gold": r["class_label"],
                                    "pred": pred, "correct": pred == r["class_label"],
                                    "logprobs": lp, "n_segments": len(sv),
                                    "n_segments_native": len(segs)},
                                   ensure_ascii=False) + "\n")
                n += 1
            if (i + 1) % 50 == 0:
                print(f"  mmpsy {i+1}/{len(rows)}", flush=True)
    return n


# ----------------------------------------------------------------- ptsd (H5)
def task_ptsd(score, out, seed):
    import pyarrow.parquet as pq
    from shortcut_harness.media import strip_placeholders
    MARK = "Above is a video"
    COLS = ["dataset", "problem", "answer", "class_label"]
    SURF = {"ptsd": "ptsd", "no_ptsd": "no ptsd"}
    rows = []
    for i in [0, 1, 2, 3]:
        t = pq.read_table(f"{ROOT}/data/raw/HBA-test/test-{i:05d}-of-00060.parquet",
                          columns=COLS).to_pandas()
        rows += [r for _, r in t[t["dataset"] == "ptsd_in_the_wild"].iterrows()]
    template = MARK + rows[0]["problem"].split(MARK, 1)[1]

    def tr(p):
        return "\n".join(l for l in p.split(MARK)[0].split("\n")
                         if l.strip() not in ("<video>", "<audio>")).strip()

    pools, pool_ids = {}, set()
    for lab, key in [("no_ptsd", "ctx_vlog"), ("ptsd", "ctx_testimony")]:
        cand = sorted([(i, r) for i, r in enumerate(rows) if r["class_label"] == lab],
                      key=lambda x: -len(tr(x[1]["problem"]).split()))
        ch = []
        for i, r in cand[:2]:
            pool_ids.add(i); ch.append(" ".join(tr(r["problem"]).split()[:60]))
        pools[key] = " ".join(ch)
    nw = min(len(p.split()) for p in pools.values())
    pools = {k: " ".join(v.split()[:nw]) for k, v in pools.items()}

    labels = ["ptsd", "no_ptsd"]
    surf = [SURF[l] for l in labels]; back = {SURF[l]: l for l in labels}
    q = strip_placeholders(template, keep_audio=False, keep_video=False, keep_text=True)
    n = 0
    with open(out, "w") as f:
        for i, r in enumerate(rows):
            if i in pool_ids:
                continue
            w = tr(r["problem"]).split()
            own = " ".join(w[:2400])
            arms = {"native": own,
                    "ctx_vlog": pools["ctx_vlog"] + " " + own,
                    "ctx_testimony": pools["ctx_testimony"] + " " + own}
            for arm, text in arms.items():
                sc = score(f"{text}\n{q}", surf)
                lp = {back[k]: v["mean"] for k, v in sc.items()}
                pred = max(lp, key=lp.get)
                f.write(json.dumps({"idx": i, "arm": arm, "gold": r["class_label"],
                                    "pred": pred, "correct": pred == r["class_label"],
                                    "logprobs": lp, "truncated": len(w) > 2400}) + "\n")
                n += 1
            if (i + 1) % 20 == 0:
                print(f"  ptsd {i+1}/{len(rows)}", flush=True)
    return n


# ------------------------------------------------------------------ Irf (H6)
def task_irf(score, out, seed, n_items=400):
    from run_h6 import TEMPLATE, SURFACE, POOL_ITEMS, POOL_WORDS, ARMS
    rows = [json.loads(l) for l in open(f"{ROOT}/data/processed/text_manifest.jsonl")]
    irf = [r for r in rows if r["source_dataset"] == "Irf" and r.get("label")]
    mwd = [r for r in rows if r["source_dataset"] == "MultiWD" and r.get("text")]
    pools, pool_uids = {}, set()

    def build(c, key):
        c = sorted(c, key=lambda r: -len(r["text"].split()))[:POOL_ITEMS]
        ch = []
        for r in c:
            pool_uids.add(r["uid"])
            ch.append(" ".join(r["text"].split()[:POOL_WORDS // POOL_ITEMS]))
        pools[key] = " ".join(ch)
    build([r for r in irf if r["label"] == "yes"], "ctx_yes")
    build([r for r in irf if r["label"] == "no"], "ctx_no")
    build(mwd, "ctx_neutral")
    nt = min(len(p.split()) for p in pools.values())
    pools = {k: " ".join(v.split()[:nt]) for k, v in pools.items()}

    ev = [r for r in irf if r["uid"] not in pool_uids]
    random.Random(seed).shuffle(ev)
    ev = ev[:n_items]
    labels = ["yes", "no"]
    surf = [SURFACE[l] for l in labels]; back = {SURFACE[l]: l for l in labels}
    n = 0
    with open(out, "w") as f:
        for k, r in enumerate(ev):
            own = " ".join(r["text"].split()[:900])
            for arm in ARMS:
                text = own if arm == "native" else pools[arm] + " " + own
                sc = score(TEMPLATE.format(text=text, question=r["question"]), surf)
                lp = {back[kk]: v["mean"] for kk, v in sc.items()}
                pred = max(lp, key=lp.get)
                f.write(json.dumps({"uid": r["uid"], "arm": arm, "gold": r["label"],
                                    "question": r["question"], "pred": pred,
                                    "correct": pred == r["label"], "logprobs": lp}) + "\n")
                n += 1
            if (k + 1) % 100 == 0:
                print(f"  irf {k+1}/{len(ev)}", flush=True)
    return n


TASKS = {"mmpsy": task_mmpsy, "ptsd": task_ptsd, "irf": task_irf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = (f"{ROOT}/experiments/h7-full-matrix/results/"
           f"{args.task}_{args.model}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    score = make_scorer(args.model, args.device)
    print(f"loaded {args.model} in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    n = TASKS[args.task](score, out, args.seed)
    print(f"wrote {out}  ({n} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
