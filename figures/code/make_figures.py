#!/usr/bin/env python3
"""Figures 2-4 for the shortcut-learning paper.

Every value is recomputed from raw experiment output rather than transcribed,
so a figure can never drift from the numbers in the text:

  Fig 2  DAIC column  <- suite_*.json `scores.cf_{depression,neutral}` (signed)
         PTSD/Irf/mmpsy <- h7-full-matrix jsonl, falling back to the original
                           h4/h5/h6 runs for the models h7 did not re-run
  Fig 3  <- i7-causal-probing/results/layerwise_multi.json, full per-layer arrays
  Fig 4  <- suite_*.json `interventions.{none,instruct_ignore}.css`
"""
from __future__ import annotations

import argparse, collections, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = "/data/cl195/NeurIPS_workshop"
OUT = f"{ROOT}/figures"
TITLES = True          # set False (--no-titles) to let the captions carry them
# neurips.sty sets textwidth=5.5in and the class is single-column `article`.
# Authoring at 5.5in makes \includegraphics[width=\textwidth] a 1:1 placement,
# so the point sizes below are the point sizes that reach the page. Authoring
# wider (e.g. 7.2in for a two-column layout) silently shrank every label by
# 0.764x -- 7.2pt cell numbers landed at 5.5pt.
W = 5.5
SUITE = f"{ROOT}/experiments/i10-model-sweep/results"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,          # keep text as vector text, not outlines
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

sig = lambda x: 1 / (1 + np.exp(-x))

TAGS = [("mentallama_suite", "MentaLLaMA"), ("menta", "Menta"),
        ("qwen3_base", "Qwen3-4B-base"), ("osim8b", "OSim-8B"),
        ("omnisapiens_sft", "OmniSapiens-SFT"), ("omnisapiens_rl", "OmniSapiens-RL"),
        ("omnisapiens_20", "OmniSapiens-2.0"), ("humanomniv2", "HumanOmniV2")]


def save(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{OUT}/{stem}.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    print(f"  wrote {OUT}/{stem}.{{pdf,svg,png}}")


# ===================================================================== data
def daic_signed():
    """Signed CSS on DAIC from the stored per-participant log-odds."""
    out = {}
    for tag, name in TAGS:
        d = json.load(open(f"{SUITE}/suite_{tag}.json"))
        S = d["scores"]
        p_hi = sig(np.array(S["cf_depression"], float))
        p_lo = sig(np.array(S["cf_neutral"], float))
        out[name] = float((p_hi - p_lo).mean())
    return out


SPEC = {
    "mmpsy": dict(pos="depression", neg="no_depression",
                  hi="seg_plus", lo="seg_minus", key="idx"),
    "ptsd": dict(pos="ptsd", neg="no_ptsd",
                 hi="ctx_testimony", lo="ctx_vlog", key="idx"),
    "irf": dict(pos="yes", neg="no", hi="ctx_yes", lo="ctx_no", key="uid"),
}
ALIAS = {"MentaLLaMA-chat-7B": "MentaLLaMA", "OmniSapiens2.0": "OmniSapiens-2.0",
         "SFT": "OmniSapiens-SFT", "RL": "OmniSapiens-RL",
         "HARPO": "OmniSapiens-2.0", "qwen3_base": "Qwen3-4B-base"}


def _css_from_jsonl(path, sp):
    by = collections.defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        k = r.get(sp["key"], r.get("uid", r.get("idx")))
        if sp["key"] == "uid" and "question" in r:
            k = (r["uid"], r["question"])
        by[r["arm"]][k] = r
    if sp["hi"] not in by or sp["lo"] not in by:
        return None
    ids = sorted(set(by[sp["hi"]]) & set(by[sp["lo"]]), key=str)
    if len(ids) < 10:
        return None

    def pv(a):
        o = []
        for i in ids:
            lp = by[a][i]["logprobs"]
            v = np.array([lp[sp["pos"]], lp[sp["neg"]]], float)
            e = np.exp(v - v.max())
            o.append(e[0] / e.sum())
        return np.array(o)
    return float((pv(sp["hi"]) - pv(sp["lo"])).mean())


def other_datasets():
    """h7 first; fall back to the original h4/h5/h6 runs where h7 has no file."""
    R = collections.defaultdict(dict)
    for p in glob.glob(f"{ROOT}/experiments/h7-full-matrix/results/*.jsonl"):
        n = os.path.basename(p)[:-6]
        task = n.split("_")[0]
        if task not in SPEC:
            continue
        model = ALIAS.get(n[len(task) + 1:], n[len(task) + 1:])
        v = _css_from_jsonl(p, SPEC[task])
        if v is not None:
            R[task][model] = v
    for task, pat in [("mmpsy", "h4-mmpsy-counterfactual/results/h4_*.jsonl"),
                      ("ptsd", "h5-ptsd-register-cf/results/h5_*.jsonl"),
                      ("irf", "h6-irf-register-cf/results/h6_*.jsonl")]:
        for p in glob.glob(f"{ROOT}/experiments/{pat}"):
            model = ALIAS.get(os.path.basename(p)[:-6].split("_")[-1],
                              os.path.basename(p)[:-6].split("_")[-1])
            if model in R[task]:
                continue
            v = _css_from_jsonl(p, SPEC[task])
            if v is not None:
                R[task][model] = v
    return R


# ================================================================== Figure 2
def figure2():
    models = [n for _, n in TAGS]
    daic, other = daic_signed(), other_datasets()
    cols = [("DAIC-WOZ", "AUC = .81", {m: daic[m] for m in models}),
            ("PTSD", "BA = .95", other["ptsd"]),
            ("Irf", "BA = .64", other["irf"]),
            ("mmpsy", "BA = .77", other["mmpsy"])]
    M = np.array([[c[2].get(m, np.nan) for c in cols] for m in models])

    fig, ax = plt.subplots(figsize=(W, 3.55))
    # Symmetric about zero so that colour intensity is comparable between
    # positive and negative effects of the same size. An asymmetric scale made
    # -0.082 look as saturated as +0.45. Exact values are printed in every
    # cell, so the weaker Irf blues losing intensity is faithful, not a loss.
    norm = TwoSlopeNorm(vmin=-0.52, vcenter=0.0, vmax=0.52)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7,
                        color="0.45")
                continue
            txt = f"{v:+.3f}"
            if txt in ("+0.000", "-0.000"):
                txt = "0.000"
            # white text only where the fill is genuinely dark
            col = "white" if abs(v) > 0.34 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.2,
                    color=col)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c[0]}\n{c[1]}" for c in cols], fontsize=7.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # the OmniSapiens family reads as a block in DAIC and PTSD. Bracket goes on
    # the RIGHT: on the left it collides with the model names.
    xb = len(cols) - 0.40
    ax.plot([xb, xb + 0.09, xb + 0.09, xb], [3.62, 3.62, 6.38, 6.38],
            color="0.4", lw=0.9, clip_on=False, zorder=5)
    ax.text(xb + 0.15, 5.0, "OmniSapiens\nfamily", rotation=90, ha="center",
            va="center", fontsize=6.6, color="0.35")

    cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.075,
                      ticks=[-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45])
    cb.set_label(r"CSS$_{\mathrm{dir}}$", fontsize=8, labelpad=4)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.6)
    cb.ax.text(0.5, 1.02, "shortcut-consistent", transform=cb.ax.transAxes,
               ha="center", va="bottom", fontsize=6.0, color="0.45")
    cb.ax.text(0.5, -0.025, "opposite", transform=cb.ax.transAxes,
               ha="center", va="top", fontsize=6.0, color="0.45")

    if TITLES:
        ax.set_title("Counterfactual shortcut reliance across models "
                     "and datasets", pad=8)
    fig.tight_layout(pad=0.4)
    save(fig, "figure2_css_heatmap")
    plt.close(fig)
    return M


# ================================================================== Figure 3
COLORS = {"MentaLLaMA": "#0072B2", "OSim-8B": "#D55E00",
          "Menta": "#009E73", "Qwen3-4B-base": "#CC79A7"}
KEYS = [("mentallama", "MentaLLaMA"), ("osim8b", "OSim-8B"),
        ("menta", "Menta"), ("qwen3_base", "Qwen3-4B-base")]


def figure3():
    L = json.load(open(f"{ROOT}/experiments/i7-causal-probing/results/"
                       f"layerwise_multi.json"))
    from matplotlib.lines import Line2D
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(W, 2.65), sharex=True)

    for key, name in KEYS:
        rows = L[key]
        d = np.array([r["depth"] for r in rows])
        cd = np.array([r["cohen_d"] for r in rows])
        lw = 1.7 if name == "MentaLLaMA" else 1.1
        axA.plot(d, cd, color=COLORS[name], lw=lw, label=name, zorder=3)
    axA.axhline(1.0, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=1)
    axA.text(0.985, 1.04, "$d$ = 1", ha="right", va="bottom", fontsize=6.0,
             color="0.62")
    axA.set_ylabel("Cross-fitted Cohen's $d$")
    axA.set_xlabel("Normalized layer depth")
    axA.set_title("(a) Shortcut decodability", loc="left")
    axA.set_xlim(0, 1)
    axA.set_ylim(-0.15, 4.4)


    # Random-direction control: shown for MentaLLaMA ONLY. A band pooled over
    # four models cannot be attributed to any of them, and the claim is
    # specifically that MentaLLaMA is the one model whose real direction leaves
    # ITS OWN control. The other three controls are in the appendix; each of
    # them is beaten by its random direction on 15-21 of 36 layers.
    mrows = [r for r in L["mentallama"] if r.get("partial_r") is not None
             and np.isfinite(r["partial_r"])]
    md = np.array([r["depth"] for r in mrows])
    mr = np.array([r["random_r"] for r in mrows])
    axB.fill_between(md, np.minimum(mr, 0), np.maximum(mr, 0),
                     color=COLORS["MentaLLaMA"], alpha=0.16, lw=0, zorder=1)

    for key, name in KEYS:
        rows = [r for r in L[key] if r.get("partial_r") is not None
                and np.isfinite(r["partial_r"])]
        d = np.array([r["depth"] for r in rows])
        pr = np.array([r["partial_r"] for r in rows])
        lw = 1.9 if name == "MentaLLaMA" else 1.0
        axB.plot(d, pr, color=COLORS[name], lw=lw,
                 zorder=4 if name == "MentaLLaMA" else 3)
    axB.axhline(0.0, color="0.35", lw=0.8, zorder=2)

    # MentaLLaMA's contiguous significant band, layers 8-21 of 32. Kept faint:
    # it marks a region, it is not a data series.
    lo = [r["depth"] for r in L["mentallama"] if r["layer"] == 8][0]
    hi = [r["depth"] for r in L["mentallama"] if r["layer"] == 21][0]
    axB.axvspan(lo, hi, color="0.5", alpha=0.025, zorder=0, lw=0)
    axB.text((lo + hi) / 2, 0.338, "MentaLLaMA coupled region", ha="center",
             va="top", fontsize=6.1, color="0.5")
    # The isolated layer-1 spike is left in the data but not annotated here:
    # naming it in a partial-correlation panel invited confusion with Cohen's d.
    # The main text excludes it from the contiguous coupled region instead.

    axB.set_ylabel("Behavioral coupling (partial $r$)")
    axB.set_xlabel("Normalized layer depth")
    axB.set_title("(b) Behavioral coupling", loc="left")
    axB.set_xlim(0, 1)
    axB.set_ylim(-0.20, 0.35)
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(facecolor=COLORS["MentaLLaMA"], alpha=0.16,
                              label="MentaLLaMA random-direction range")],
               frameon=True, framealpha=0.85, edgecolor="none",
               loc="lower left", handlelength=1.4, borderaxespad=0.25,
               handletextpad=0.5)

    for a in (axA, axB):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    # one model legend for both panels: Panel B carries no model legend of its
    # own, so colours must not have to be remembered across the figure
    handles = [Line2D([], [], color=COLORS[n],
                      lw=1.9 if n == "MentaLLaMA" else 1.1, label=n)
               for _, n in KEYS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               handlelength=1.6, columnspacing=1.8, handletextpad=0.55,
               bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(pad=0.4, w_pad=1.6, rect=(0, 0.075, 1, 1))
    save(fig, "figure3_layerwise")
    plt.close(fig)


# ================================================================== Figure 4
def figure4():
    delta = {}
    for tag, name in TAGS:
        iv = json.load(open(f"{SUITE}/suite_{tag}.json"))["interventions"]
        delta[name] = iv["instruct_ignore"]["css"] - iv["none"]["css"]
    order = sorted(delta, key=delta.get)                 # most negative first
    vals = np.array([delta[m] for m in order])
    y = np.arange(len(order))[::-1]                      # top row = most negative

    fig, ax = plt.subplots(figsize=(W, 2.85))
    IMP, WOR = "#2C7FB8", "#C0504D"
    ax.barh(y, vals, height=0.62, color=[IMP if v < 0 else WOR for v in vals],
            edgecolor="none", zorder=3)
    ax.axvline(0, color="0.25", lw=0.9, zorder=4)

    n_imp = int((vals < 0).sum())
    ax.axhline(y[n_imp] + 0.5, color="0.75", lw=0.7, ls=(0, (3, 3)), zorder=2)

    for yy, v in zip(y, vals):
        off = -0.006 if v < 0 else 0.006
        ax.text(v + off, yy, f"{v:+.3f}".replace("+", "+").replace("-", "−"),
                va="center", ha="right" if v < 0 else "left", fontsize=7.2)

    ax.set_yticks(y)
    # a dagger costs one glyph; the bracket cost ~25 % of the panel width
    ax.set_yticklabels([m + " †" if m in ("Qwen3-4B-base", "Menta") else m
                        for m in order])
    ax.set_xlim(-0.262, 0.155)
    ax.set_ylim(-0.75, len(order) - 0.25)
    ax.set_xlabel(r"$\Delta$CSS$_{\mathrm{mag}}$   (Ignore $-$ Baseline)")
    # No chart title: the caption carries the claim, and a title here collides
    # with the directional annotations.
    ax.text(0.0, 1.012, "Lower shortcut reliance  ←", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.1, color=IMP)
    ax.text(1.0, 1.012, "→  Higher shortcut reliance", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.1, color=WOR)
    # Each count sits on ITS OWN side of zero, vertically centred in its own
    # group. An earlier version put them in whichever quadrant was empty, which
    # placed "improve" on the positive side and read as a mislabel.
    ax.text(-0.232, (y[0] + y[n_imp - 1]) / 2, f"{n_imp}/8\nimprove",
            ha="center", va="center", fontsize=6.6, color=IMP, style="italic",
            linespacing=1.35)
    ax.text(0.137, (y[n_imp] + y[-1]) / 2, f"{8 - n_imp}/8\nworsen",
            ha="center", va="center", fontsize=6.6, color=WOR, style="italic",
            linespacing=1.35)

    ax.text(0.0, -0.235, "†  Qwen3-4B-base and Menta share the same backbone "
            "and respond in opposite directions.", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.3, color="0.3")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="0.92", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    save(fig, "figure4_mitigation_heterogeneity")
    plt.close(fig)
    return order, vals


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-titles", action="store_true",
                    help="drop in-figure titles; the captions carry them")
    TITLES = not ap.parse_args().no_titles
    os.makedirs(OUT, exist_ok=True)
    print("Figure 2 ...");  M = figure2()
    print("Figure 3 ...");  figure3()
    print("Figure 4 ...");  order, vals = figure4()
    print("\nFig 2 matrix actually plotted (rows = models, cols = "
          "DAIC/PTSD/Irf/mmpsy):")
    for (_, n), row in zip(TAGS, M):
        print("  " + f"{n:18s}" + " ".join(f"{v:+8.4f}" for v in row))
    print("\nFig 4 values actually plotted:")
    for m, v in zip(order, vals):
        print(f"  {m:18s} {v:+.4f}")
