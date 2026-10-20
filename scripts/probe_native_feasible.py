#!/usr/bin/env python3
"""Which MELD items can `native` actually be scored on?

Native resolution costs ~371 input tokens per frame, so a long clip blows past
Qwen2.5-Omni's 32768 position limit -- transformers only warns, then produces
garbage. Native is therefore not a valid reference point on every item, and a
calibration curve anchored to it must be restricted to the items where it is
sound. CPU only; no model needed.
"""
import io, json

import av
import pyarrow.parquet as pq

ROOT = "/data/cl195/NeurIPS_workshop"
SHARD = f"{ROOT}/data/raw/HBA-test/test-00005-of-00060.parquet"
TOK_PER_FRAME = 371          # calibrated: a 54-frame clip measured 20,063 tokens
LIMIT, MARGIN = 32768, 30000


def main():
    t = pq.ParquetFile(SHARD).read().to_pydict()
    rows = [i for i, d in enumerate(t["dataset"])
            if d == "meld_emotion" and t["videos"][i]]

    frames = []
    for i in rows:
        c = av.open(io.BytesIO(bytes(t["videos"][i][0])))
        frames.append((i, sum(1 for _ in c.decode(video=0))))
        c.close()

    n = sorted(f for _, f in frames)
    q = lambda p: n[int(p * (len(n) - 1))]
    print(f"{len(rows)} meld_emotion clips with video")
    print("frames        p10=%d  median=%d  p90=%d  max=%d"
          % (q(.1), q(.5), q(.9), n[-1]))
    print("native tokens p10=%d  median=%d  p90=%d  max=%d"
          % (q(.1) * TOK_PER_FRAME, q(.5) * TOK_PER_FRAME,
             q(.9) * TOK_PER_FRAME, n[-1] * TOK_PER_FRAME))

    ok = [i for i, f in frames if f * TOK_PER_FRAME < MARGIN]
    print(f"\nposition limit {LIMIT}, working margin {MARGIN}")
    print("  max scoreable at native: %d frames (~%.1fs at 27fps)"
          % (MARGIN // TOK_PER_FRAME, (MARGIN // TOK_PER_FRAME) / 27))
    print("  native-feasible: %d / %d clips (%.0f%%)"
          % (len(ok), len(rows), 100 * len(ok) / len(rows)))

    lab = [t["class_label"][i] for i in ok]
    from collections import Counter
    print("  label dist of feasible subset:", dict(Counter(lab).most_common()))

    out = f"{ROOT}/data/processed/calib_rows_native_feasible.json"
    json.dump({"shard": SHARD, "tok_per_frame": TOK_PER_FRAME,
               "margin": MARGIN, "rows": ok,
               "frames": {str(i): f for i, f in frames}}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
