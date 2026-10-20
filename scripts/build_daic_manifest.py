#!/usr/bin/env python3
"""Index the fetched DAIC-WOZ files into a single manifest.

Emits data/processed/daic_manifest.jsonl with one row per participant:
labels, split, absolute media paths, transcript stats.  Purely an index --
no task framing or prompt construction here.
"""
import json, os, wave

import pandas as pd

BASE = "/data/cl195/NeurIPS_workshop/data/raw/daic_woz"
OUT = "/data/cl195/NeurIPS_workshop/data/processed"

SPLITS = {
    "train": ("train_split_Depression_AVEC2017.csv", "PHQ8_Binary", "PHQ8_Score"),
    "dev": ("dev_split_Depression_AVEC2017.csv", "PHQ8_Binary", "PHQ8_Score"),
    "test": ("full_test_split.csv", "PHQ_Binary", "PHQ_Score"),
}


def wav_seconds(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def read_transcript(path):
    """A handful of transcripts carry cp1252 bytes (e.g. 0xb0) rather than UTF-8,
    and a few have stray extra tab-separated fields.  Returns (df, encoding)."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc,
                             engine="python", on_bad_lines="skip")
            return df, enc
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError(f"cannot decode {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for split, (fname, bincol, scorecol) in SPLITS.items():
        df = pd.read_csv(f"{BASE}/labels/{fname}")
        df.columns = [c.strip() for c in df.columns]
        for _, r in df.iterrows():
            pid = int(r["Participant_ID"])
            audio = f"{BASE}/audio/{pid}_AUDIO.wav"
            trans = f"{BASE}/transcript/{pid}_TRANSCRIPT.csv"
            has_t = os.path.exists(trans)

            rec = {
                "uid": f"daic/{split}/{pid}",
                "participant_id": pid,
                "split": split,
                "phq8_binary": int(r[bincol]),
                "phq8_score": int(r[scorecol]),
                "gender": int(r["Gender"]),
                "audio_path": audio if os.path.exists(audio) else None,
                "transcript_path": trans if has_t else None,
            }
            if rec["audio_path"]:
                rec["audio_seconds"] = round(wav_seconds(audio), 1)
            if has_t:
                t, enc = read_transcript(trans)
                t.columns = [c.strip() for c in t.columns]
                rec["transcript_encoding"] = enc
                t["value"] = t["value"].fillna("").astype(str)
                pt = t[t["speaker"] == "Participant"]
                rec["n_turns"] = int(len(t))
                rec["n_participant_turns"] = int(len(pt))
                rec["participant_words"] = int(pt["value"].str.split().str.len().sum())
            rows.append(rec)

    path = f"{OUT}/daic_manifest.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    d = pd.DataFrame(rows)
    print(f"wrote {path}  ({len(d)} participants)")
    print(d.groupby("split").agg(
        n=("uid", "size"),
        pos=("phq8_binary", "sum"),
        audio_ok=("audio_path", lambda s: s.notna().sum()),
        trans_ok=("transcript_path", lambda s: s.notna().sum()),
        med_min=("audio_seconds", lambda s: round(s.median() / 60, 1)),
        med_words=("participant_words", "median"),
    ).to_string())
    print(f"\ntotal audio hours: {d['audio_seconds'].sum()/3600:.1f}")


if __name__ == "__main__":
    main()
