#!/usr/bin/env python3
"""Fetch only the evaluation/test portions of the benchmarks these models
were tested on.

IntentBench, Daily-Omni and WorldSense are evaluation-only releases -- their
whole content is the test set, so there is no split to filter. The real
filtering happens for osim-post-training (test/ subdir only) and HBA
(test-*.parquet shards only, dropping the 355GB train and 23GB validation).
"""
import os, sys, time

from huggingface_hub import snapshot_download

DEST = "/data/cl195/NeurIPS_workshop/data/raw"

JOBS = {
    # HumanOmniV2 official benchmarks -- evaluation-only, take everything
    "IntentBench": dict(repo="PhilipC/IntentBench", allow=None, ignore=None),
    "Daily-Omni": dict(repo="liarliar/Daily-Omni", allow=None, ignore=None),
    # skip asset/*: those are paper figures, not data
    "WorldSense": dict(repo="honglyhly/WorldSense", allow=None,
                       ignore=["asset/*"]),
    # OSim-8B: test split only
    "osim-post-test": dict(repo="cmu-lti/osim-post-training",
                           allow=["test/*", "README.md"], ignore=None),
    # OmniSapiens2.0 / HBA: test shards only
    "HBA-test": dict(repo="HumanBehaviorAtlas/human_behavior_atlas",
                     allow=["test-*.parquet", "*.json", "README.md"],
                     ignore=None),
}


def main():
    for name in (sys.argv[1:] or JOBS):
        j = JOBS[name]
        out = os.path.join(DEST, name)
        t0 = time.time()
        print(f"[{name}] {j['repo']} -> {out}", flush=True)
        p = snapshot_download(
            repo_id=j["repo"], repo_type="dataset", local_dir=out,
            allow_patterns=j["allow"], ignore_patterns=j["ignore"],
            max_workers=8)
        sz = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(p) for f in fs)
        print(f"[{name}] done {sz/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
