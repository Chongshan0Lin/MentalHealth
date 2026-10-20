import os, sys, time
from huggingface_hub import snapshot_download
DEST="/data/cl195/NeurIPS_workshop/checkpoints"
JOBS={"OmniSapiens-SFT":"HumanBehaviorAtlas/omnisapiens_sft",
      "OmniSapiens-RL":"ddvd233/OmniSapiens-7B-RL"}
for n in (sys.argv[1:] or JOBS):
    out=os.path.join(DEST,n); t0=time.time()
    print(f"[{n}] {JOBS[n]} -> {out}", flush=True)
    p=snapshot_download(JOBS[n], local_dir=out,
        ignore_patterns=["*.bin","*.pth","*.gguf","*.msgpack","*.h5"], max_workers=8)
    sz=sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(p) for f in fs)
    print(f"[{n}] done {sz/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)
