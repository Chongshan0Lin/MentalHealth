#!/usr/bin/env python3
"""Build the Experiment-A text conditions and mine the Ellie prompt inventory.

Text conditions (protocol-independent, built now):
  full              every turn, both speakers
  participant_only  participant turns only            -> C
  ellie_only        interviewer turns only            -> S
  ellie_masked      Ellie turns replaced by [Q], turn order and count kept
                    -> separates the *lexical content* of the prompts from the
                       *structure* of the interview

Audio slicing is exposed as `speaker_segments()` but deliberately not run here:
the windowing protocol (how to fit a ~15 min interview through a 300 s Whisper
front-end) is a methodological decision that must be fixed once and applied
identically to every condition, or the ablation measures windowing instead of
modality.

Also emits a candidate Ellie prompt inventory for A2. The two pools
(depression-associated vs neutral) are seeded by keyword and MUST be
hand-verified before use.
"""
import json, re, collections

import pandas as pd

ROOT = "/data/cl195/NeurIPS_workshop"
MANIFEST = f"{ROOT}/data/processed/daic_manifest.jsonl"
OUT = f"{ROOT}/data/processed"

# seeds for the A2 prompt pools -- a starting point for manual review, not a
# finished taxonomy
DEPRESSION_SEEDS = [
    "depress", "diagnos", "therapy", "therapist", "ptsd", "medication",
    "counsel", "suicid", "sleep", "regret", "guilt", "argument", "fight",
    "close to", "shut down", "last time you felt really happy",
]
NEUTRAL_SEEDS = [
    "where are you from", "hometown", "los angeles", "travel", "weekend",
    "hobby", "hobbies", "for fun", "your job", "study", "school",
    "how are you doing today", "easy to get a good night", "dream job",
]


def read_transcript(path):
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, sep="\t", encoding=enc,
                               engine="python", on_bad_lines="skip")
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError(path)


def conditions(df):
    """-> {condition_name: rendered transcript text}"""
    df = df.copy()
    df["value"] = df["value"].fillna("").astype(str)
    df["speaker"] = df["speaker"].astype(str).str.strip()

    def render(rows):
        return "\n".join(f"{s}: {v}" for s, v in rows if v.strip())

    turns = list(zip(df["speaker"], df["value"]))
    return {
        "full": render(turns),
        "participant_only": render([(s, v) for s, v in turns
                                    if s == "Participant"]),
        "ellie_only": render([(s, v) for s, v in turns if s == "Ellie"]),
        "ellie_masked": render([(s, v if s == "Participant" else "[Q]")
                                for s, v in turns]),
    }


def speaker_segments(df, speaker):
    """(start, stop) spans in seconds for one speaker.

    Feeding these to a slicer yields speaker-isolated audio.  DAIC audio is a
    single mixed channel, so boundary cross-talk is unavoidable; pad every
    condition identically so the bias is shared rather than differential.
    """
    d = df[df["speaker"].astype(str).str.strip() == speaker]
    return [(float(a), float(b))
            for a, b in zip(d["start_time"], d["stop_time"])
            if pd.notna(a) and pd.notna(b)]


def main():
    recs = [json.loads(l) for l in open(MANIFEST)]
    rows, ellie_turns = [], collections.Counter()

    for r in recs:
        if not r.get("transcript_path"):
            continue
        df = read_transcript(r["transcript_path"])
        c = conditions(df)
        rows.append({
            "uid": r["uid"], "participant_id": r["participant_id"],
            "split": r["split"], "phq8_binary": r["phq8_binary"],
            "phq8_score": r["phq8_score"], "gender": r["gender"],
            "audio_path": r["audio_path"],
            "n_participant_spans": len(speaker_segments(df, "Participant")),
            "n_ellie_spans": len(speaker_segments(df, "Ellie")),
            **{f"text_{k}": v for k, v in c.items()},
        })
        for v in df[df["speaker"].astype(str).str.strip() == "Ellie"]["value"]:
            t = re.sub(r"\s+", " ", str(v)).strip().lower()
            if t and t != "nan":
                ellie_turns[t] += 1

    path = f"{OUT}/daic_conditions.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    d = pd.DataFrame(rows)
    print(f"wrote {path}  ({len(d)} participants)")
    for c in ["full", "participant_only", "ellie_only", "ellie_masked"]:
        w = d[f"text_{c}"].str.split().str.len()
        print("  %-18s median %5.0f words   min %4d   max %5d"
              % (c, w.median(), w.min(), w.max()))

    # --- A2 candidate prompt pools -------------------------------------
    def bucket(seeds):
        return [(t, n) for t, n in ellie_turns.most_common()
                if any(s in t for s in seeds)]

    dep, neu = bucket(DEPRESSION_SEEDS), bucket(NEUTRAL_SEEDS)
    inv = {
        "n_distinct_ellie_turns": len(ellie_turns),
        "note": "SEEDED BY KEYWORD - REQUIRES MANUAL VERIFICATION BEFORE USE",
        "depression_associated_candidates": [
            {"text": t, "n_participants": n} for t, n in dep[:120]],
        "neutral_candidates": [
            {"text": t, "n_participants": n} for t, n in neu[:120]],
    }
    ipath = f"{OUT}/daic_ellie_prompt_inventory.json"
    with open(ipath, "w") as f:
        json.dump(inv, f, indent=2)

    print(f"\nwrote {ipath}")
    print(f"  distinct Ellie turns: {len(ellie_turns)}")
    print(f"  depression-associated candidates: {len(dep)}")
    print(f"  neutral candidates:               {len(neu)}")
    print("\n  most common depression-associated (n = participants asked):")
    for t, n in dep[:8]:
        print(f"    {n:4d}  {t[:88]}")
    print("\n  most common neutral:")
    for t, n in neu[:8]:
        print(f"    {n:4d}  {t[:88]}")


if __name__ == "__main__":
    main()
