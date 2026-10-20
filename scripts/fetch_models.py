#!/usr/bin/env python3
"""Download the day-1 checkpoints into $PROJ/checkpoints.

Skips the Menta GGUF-style extras: we only take what transformers needs.
"""
import os, sys, time
from huggingface_hub import snapshot_download

DEST = "/data/cl195/NeurIPS_workshop/checkpoints"

MODELS = {
    "OmniSapiens2.0": dict(repo_id="HumanBehaviorAtlas/OmniSapiens2.0"),
    "HumanOmniV2": dict(repo_id="PhilipC/HumanOmniV2"),
    "MentaLLaMA-chat-7B": dict(repo_id="klyang/MentaLLaMA-chat-7B-hf"),
}
# never pull these; they duplicate the safetensors or are training leftovers
IGNORE = ["*.bin", "*.pth", "*.gguf", "optimizer.pt", "*.msgpack", "*.h5"]


def main():
    names = sys.argv[1:] or list(MODELS)
    for name in names:
        cfg = MODELS[name]
        out = os.path.join(DEST, name)
        t0 = time.time()
        print(f"[{name}] -> {out}", flush=True)
        p = snapshot_download(
            repo_id=cfg["repo_id"],
            local_dir=out,
            ignore_patterns=IGNORE,
            max_workers=8,
        )
        sz = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(p)
            for f in fs
        )
        print(f"[{name}] done: {sz/1e9:.1f} GB in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
