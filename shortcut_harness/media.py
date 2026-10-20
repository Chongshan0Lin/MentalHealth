#!/usr/bin/env python3
"""Media decoding for HBA rows: parquet binaries -> model-ready arrays.

⚠ Video downsampling is a methodological choice, not a performance knob.
Shrinking resolution or frame rate handicaps the video channel, and in a
conflict experiment that manufactures apparent *audio dominance* — precisely the
quantity B2 measures.  `VIDEO_PRESET` must therefore be fixed once, applied
identically to every condition and every model, and calibrated first (see
`calibrate_video_preset` in EXPERIMENT_DESIGN notes).

Measured token cost for one 2.25s 720p MELD clip (audio included):

    native 720p, all frames    20063 tokens
    720p, every 2nd frame      10443
    336p, every 2nd frame       3611
    224p, every 2nd frame       1973
    224p, every 4th frame       1028
"""
from __future__ import annotations

import io

import av
import numpy as np
import soundfile as sf
from PIL import Image

SR = 16000
PATCH = 28  # Qwen2-VL patch size; frame dims must be multiples of this

# (target_height, frame_stride). None height = keep native.
VIDEO_PRESETS = {
    "native": (None, 1),
    "half": (None, 2),
    "336p": (336, 2),
    "224p": (224, 2),
    "224p_sparse": (224, 4),
}


def load_audio(blob: bytes) -> np.ndarray:
    """WAV binary -> float32 mono @16kHz."""
    wav, sr = sf.read(io.BytesIO(blob), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32)


def _resize(frames: np.ndarray, height: int | None) -> np.ndarray:
    h0, w0 = frames.shape[1], frames.shape[2]
    if height is None:
        h, w = h0 - h0 % PATCH, w0 - w0 % PATCH
    else:
        h = height - height % PATCH
        w = int(w0 * h / h0)
        w -= w % PATCH
    if (h, w) == (h0, w0):
        return frames
    return np.stack([np.asarray(Image.fromarray(f).resize((w, h)))
                     for f in frames])


def load_video(blob: bytes, preset: str = "224p", strip: bool = True) -> np.ndarray:
    """MP4 binary -> (T,H,W,3) uint8.

    `strip=True` re-muxes to video-only first.  HBA MP4s carry their own AAC
    track; if it reaches the model, a swapped-audio conflict item holds two
    competing audio sources and R_A/R_V stop meaning anything.
    """
    if strip:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from strip_video_audio import strip_audio
        blob = strip_audio(blob)

    c = av.open(io.BytesIO(blob))
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    if not frames:
        raise ValueError("no video frames decoded")

    height, stride = VIDEO_PRESETS[preset]
    return _resize(np.stack(frames[::stride]), height)


def row_media(row: dict, idx: int = 0, want_audio=True, want_video=True,
              preset: str = "224p") -> dict:
    """Pull one HBA parquet row's media into a scorer-ready dict.

    `row` is a single row (dict-like); `idx` selects among the media attached to
    that row, and every HBA row carries at most one of each. The columns are
    `list<binary>`, so `row["videos"][idx]` is already the MP4 bytes -- an
    earlier version indexed one level deeper (`[idx][0]`), which subscripts the
    bytes object and yields an int, and av then failed with InvalidDataError.
    """
    out = {"audio": None, "video": None}
    if want_audio and len(row["audios"]):
        out["audio"] = load_audio(bytes(row["audios"][idx]))
    if want_video and len(row["videos"]):
        out["video"] = load_video(bytes(row["videos"][idx]), preset=preset)
    return out


def strip_placeholders(problem: str, keep_audio: bool, keep_video: bool,
                       keep_text: bool = True) -> str:
    """Drop the <audio>/<video> markers (and optionally the transcript line)
    so the prompt matches the media actually supplied.

    HBA `problem` fields open with the placeholders on their own lines,
    followed by the transcript, followed by the question.
    """
    lines = problem.split("\n")
    head, rest = [], []
    for ln in lines:
        (head if ln.strip() in ("<audio>", "<video>") else rest).append(ln)
    if not keep_text and len(rest) > 1:
        rest = rest[1:]          # first non-placeholder line is the transcript
    kept = ([l for l in head if l.strip() == "<video>"] if keep_video else []) \
         + ([l for l in head if l.strip() == "<audio>"] if keep_audio else [])
    return "\n".join(kept + rest)
