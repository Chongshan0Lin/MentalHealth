import os, sys, time
from huggingface_hub import snapshot_download
DEST="/data/cl195/NeurIPS_workshop/checkpoints"
JOBS={
 "OSim-8B":        dict(repo_id="cmu-lti/osim-8b", ignore=["*.bin","*.pth","*.gguf","*.msgpack","*.h5"]),
 "Qwen3-4B-base":  dict(repo_id="Qwen/Qwen3-4B-Instruct-2507", ignore=["*.gguf"]),
 "Menta-LoRA":     dict(repo_id="mHealthAI/Menta", allow=["Menta/*"], ignore=["*.gguf","*optimizer.pt","*rng_state.pth","*scheduler.pt"]),
 "Emotion-LLaMA-v2": dict(repo_id="ooochen/Emotion-LLaMA-v2", ignore=[]),
}
for name in (sys.argv[1:] or JOBS):
    j=JOBS[name]; out=os.path.join(DEST,name); t0=time.time()
    print(f"[{name}] {j['repo_id']} -> {out}", flush=True)
    p=snapshot_download(repo_id=j["repo_id"], local_dir=out,
                        allow_patterns=j.get("allow"), ignore_patterns=j.get("ignore"),
                        max_workers=8)
    sz=sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(p) for f in fs)
    print(f"[{name}] done {sz/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)
