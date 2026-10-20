#!/usr/bin/env python3
"""Re-mux an HBA video binary to video-only.

Why this exists: the MP4s embedded in the HBA parquet carry their own AAC audio
track (streams = [h264, aac, bin_data]).  In a swapped-audio conflict item that
native track competes with the audio we deliberately paired in, and R_A / R_V
become uninterpretable.  Every conflict condition must run on stripped video.

`strip_audio(mp4_bytes) -> mp4_bytes` copies the video stream packet-for-packet
(no re-encode, so no quality change and no frame-timing drift) and drops
everything else.

Run directly to self-test against the deployed HBA test shards.
"""
import io

import av


def strip_audio(mp4_bytes: bytes) -> bytes:
    """Return the same MP4 with only its video stream, without re-encoding."""
    src = av.open(io.BytesIO(mp4_bytes))
    vin = next(s for s in src.streams if s.type == "video")

    buf = io.BytesIO()
    dst = av.open(buf, mode="w", format="mp4")
    vout = dst.add_stream_from_template(vin)

    for packet in src.demux(vin):
        if packet.dts is None:          # flush packets carry no timestamp
            continue
        packet.stream = vout
        dst.mux(packet)

    dst.close()
    src.close()
    return buf.getvalue()


def describe(mp4_bytes: bytes):
    c = av.open(io.BytesIO(mp4_bytes))
    info = {
        "streams": [s.type for s in c.streams],
        "duration_s": round(float(c.duration or 0) / 1e6, 2),
    }
    v = [s for s in c.streams if s.type == "video"]
    if v:
        info["size"] = f"{v[0].width}x{v[0].height}"
        info["frames"] = sum(1 for _ in c.decode(video=0))
    c.close()
    return info


def _self_test():
    import glob
    import pyarrow.parquet as pq

    shard = "/data/cl195/NeurIPS_workshop/data/raw/HBA-test/test-00005-of-00060.parquet"
    t = pq.ParquetFile(shard).read().to_pydict()
    rows = [i for i, d in enumerate(t["dataset"])
            if d == "meld_emotion" and t["videos"][i]][:5]

    print(f"{'row':>6} {'before':>34} {'after':>34} {'bytes':>18}")
    for i in rows:
        raw = bytes(t["videos"][i][0])
        out = strip_audio(raw)
        b, a = describe(raw), describe(out)
        assert "audio" not in a["streams"], f"row {i}: audio survived"
        assert a["frames"] == b["frames"], f"row {i}: frame count changed"
        assert abs(a["duration_s"] - b["duration_s"]) < 0.1, f"row {i}: duration drift"
        print(f"{i:>6} {str(b['streams']):>24} {b['frames']:>4}f "
              f"{str(a['streams']):>24} {a['frames']:>4}f "
              f"{len(raw)/1e3:>8.0f}->{len(out)/1e3:.0f} KB")

    print("\nvideo stream preserved, audio removed, frame count and duration intact")
    print("SELF-TEST OK")


if __name__ == "__main__":
    _self_test()
