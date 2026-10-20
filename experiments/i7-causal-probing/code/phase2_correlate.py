#!/usr/bin/env python3
"""I7 Phase 2 -- does the representational shift track the behavioural shift?

Phase 1 showed the interviewer's context shifts the representation of the
patient's own words, and that the shift is highly SHARED across patients
(cos to the mean direction ~0.9) but varies in MAGNITUDE (coefficient of
variation ~0.33). Phase 2 asks whether that variation is the thing that drives
behaviour:

    do patients whose representation is pushed harder also show a larger
    change in predicted depression probability?

    dh_i = w . ( h_clinical_i - h_neutral_i )     representational movement
    dp_i = p(dep | clinical_i) - p(dep | neutral_i)   behavioural movement

THE CONTROL THAT DECIDES WHETHER THIS MEANS ANYTHING. dh and dp come from the
same forward pass, so they correlate for trivial reasons: a patient whose
representation moved a lot will show a prediction that moved a lot regardless of
direction. The headline is therefore the PARTIAL correlation of dh with dp
controlling for total displacement ||h_clinical - h_neutral||. The raw
correlation is printed alongside but must never be quoted alone.

Two probe directions are compared:
  mean-shift direction  the average of (h_clinical - h_neutral); the direction
                        Phase 1's diagnostic showed dominates
  probe direction       the trained logistic-probe weight vector, refit here on
                        all patients (no CV needed: it is used as a direction,
                        not evaluated for accuracy)
plus a random-direction control at each layer.
"""
from __future__ import annotations

import json
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = "/data/cl195/NeurIPS_workshop"
RES = f"{ROOT}/experiments/i7-causal-probing/results"
A2B = f"{ROOT}/experiments/a2-ellie-counterfactual/results/a2b_MentaLLaMA-chat-7B.jsonl"
rng = np.random.default_rng(0)
sig = lambda x: 1 / (1 + np.exp(-x))


def partial_corr(x, y, z):
    """corr(x, y | z) via residualisation on z (with intercept)."""
    Z = np.vstack([np.ones_like(z), z]).T
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)


def boot_partial(x, y, z, B=4000):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        try:
            v.append(partial_corr(x[i], y[i], z[i])[0])
        except Exception:
            pass
    return np.percentile(v, [2.5, 97.5])


def main():
    d = np.load(f"{RES}/daic_hidden_participant.npz", allow_pickle=True)
    H, layers, arms = d["H"], d["layers"], [str(a) for a in d["arms"]]
    meta = json.load(open(f"{RES}/daic_hidden_participant_meta.json"))
    pid = np.array([m["participant_id"] for m in meta])
    i0, i1 = arms.index("cf_neutral"), arms.index("cf_depression")

    # behavioural shift, from the run whose prompts these representations match
    lp = {}
    for line in open(A2B):
        r = json.loads(line)
        lp.setdefault(r["arm"], {})[r["participant_id"]] = (
            r["logprobs"]["Yes"] - r["logprobs"]["No"])
    dp = np.array([sig(lp["cf_depression"][p]) - sig(lp["cf_neutral"][p])
                   for p in pid])
    print(f"n = {len(pid)} patients")
    print(f"behavioural shift dp: mean {dp.mean():+.4f}  sd {dp.std():.4f}  "
          f"range [{dp.min():+.3f}, {dp.max():+.3f}]\n")

    y = np.concatenate([np.zeros(len(pid)), np.ones(len(pid))]).astype(int)
    out = []
    print("=" * 96)
    print("PARTIAL correlation of representational shift with behavioural shift")
    print("(controlling for total displacement ||h_clin - h_neut||)")
    print("=" * 96)
    print(f"{'layer':>5s} {'direction':>12s} {'raw r':>8s} {'partial r':>10s} "
          f"{'[95% CI]':>18s} {'p':>9s}")
    print("-" * 96)
    for li, L in enumerate(layers):
        A, B = H[:, i0, li, :], H[:, i1, li, :]
        D = B - A
        norm = np.linalg.norm(D, axis=1)

        dirs = {}
        mu = D.mean(0); dirs["mean-shift"] = mu / np.linalg.norm(mu)
        X = np.concatenate([A, B])
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=256, random_state=0),
                             LogisticRegression(max_iter=3000, C=0.1))
        pipe.fit(X, y)
        w = (pipe[:-1].inverse_transform(
                pipe[-1].coef_.reshape(1, -1))[0]
             if hasattr(pipe[:-1], "inverse_transform") else None)
        if w is None or not np.isfinite(w).all():
            w = pipe[1].components_.T @ pipe[-1].coef_[0]
        dirs["probe"] = w / np.linalg.norm(w)
        rv = rng.normal(size=D.shape[1]); dirs["random"] = rv / np.linalg.norm(rv)

        for name, u in dirs.items():
            dh = D @ u
            raw = stats.pearsonr(dh, dp)
            pc = partial_corr(dh, dp, norm)
            lo, hi = boot_partial(dh, dp, norm)
            print(f"{int(L):5d} {name:>12s} {raw[0]:8.3f} {pc[0]:10.3f} "
                  f"[{lo:7.3f},{hi:7.3f}] {pc[1]:9.2e}")
            out.append(dict(layer=int(L), direction=name, raw_r=raw[0],
                            partial_r=pc[0], ci=[lo, hi], p=pc[1]))
        print()

    json.dump(out, open(f"{RES}/phase2_correlation.json", "w"), indent=2)
    real = [o for o in out if o["direction"] in ("mean-shift", "probe")]
    rand = [o for o in out if o["direction"] == "random"]
    sigf = lambda o: o["ci"][0] > 0 or o["ci"][1] < 0
    hit, rand_hit = [o for o in real if sigf(o)], [o for o in rand if sigf(o)]
    best_real = max((abs(o["partial_r"]) for o in real), default=0.0)
    best_rand = max((abs(o["partial_r"]) for o in rand), default=0.0)
    rate_real = len(hit) / max(len(real), 1)
    rate_rand = len(rand_hit) / max(len(rand), 1)

    print("=" * 96)
    print("PHASE 2 GATE")
    print("=" * 96)
    print(f"  real   : {len(hit)}/{len(real)} significant "
          f"(rate {rate_real:.2f}), best |partial r| = {best_real:.3f}")
    print(f"  random : {len(rand_hit)}/{len(rand)} significant "
          f"(rate {rate_rand:.2f}), best |partial r| = {best_rand:.3f}")

    # A correlation only counts if it is DIRECTION-SPECIFIC. The shift vector is
    # ~90% a single shared direction, so projecting onto ANY direction recovers
    # the per-patient magnitude and correlates with dp for reasons that have
    # nothing to do with the probe. The random control is therefore not
    # decoration -- it is the test. An earlier version of this gate checked only
    # |r| > 0.3 and would have reported a spurious "WEAK PASS".
    beats_random = best_real > 2 * best_rand and rate_real > rate_rand
    strong = best_real > 0.3

    if beats_random and strong:
        b = max(hit, key=lambda o: abs(o["partial_r"]))
        print(f"\n  PASS -- layer {b['layer']} / {b['direction']}: "
              f"partial r = {b['partial_r']:+.3f} "
              f"[{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}], "
              f"direction-specific and above the r > 0.3 bar.")
        print("  -> proceed to Phase 3 (LEACE erasure + control task).")
    elif not beats_random:
        print("\n  FAIL -- the correlation is NOT direction-specific.")
        print("  Random directions reach comparable magnitude and significance,")
        print("  so projecting onto the probe direction explains nothing that")
        print("  projecting onto an arbitrary direction does not. This follows")
        print("  from the shift being ~90% a single shared direction (Phase 1")
        print("  diagnostic): every projection recovers the same per-patient")
        print("  magnitude.")
        print("  -> do NOT run Phase 3. The context shifts the representation")
        print("     globally, but that shift does not encode which patients move.")
    else:
        print(f"\n  FAIL -- direction-specific but below the pre-committed "
              f"r > 0.3 bar (best {best_real:.3f}).")
        print("  -> do NOT run Phase 3; report as a weak association.")
    print(f"\nwrote {RES}/phase2_correlation.json")


if __name__ == "__main__":
    main()
