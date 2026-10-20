#!/usr/bin/env python3
"""Selectively fetch DAIC-WOZ members over HTTP Range.

The per-participant zips are ~480MB each but ~85% of that is <ID>_CLNF_hog.bin,
plus large CLNF landmark dumps.  We only pull AUDIO + TRANSCRIPT, which cuts the
transfer from ~90GB to ~3GB.  Range requests let us read the zip central
directory and then only the members we want.

Usage:
    python fetch_daic.py --out DIR [--members audio,transcript] [--workers 8]
"""
import argparse, io, os, re, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/"
META = [
    "train_split_Depression_AVEC2017.csv",
    "dev_split_Depression_AVEC2017.csv",
    "test_split_Depression_AVEC2017.csv",
    "full_test_split.csv",
    "DAICWOZDepression_Documentation_AVEC2017.pdf",
]
# member selectors -> substring match on the filename inside the zip
SELECTORS = {
    "audio": "_AUDIO.wav",
    "transcript": "_TRANSCRIPT.csv",
    "covarep": "_COVAREP.csv",
    "formant": "_FORMANT.csv",
    "aus": "_CLNF_AUs.txt",
    "gaze": "_CLNF_gaze.txt",
    "pose": "_CLNF_pose.txt",
}
# where each selector's files get written, relative to --out
SUBDIR = {"audio": "audio", "transcript": "transcript"}


class HTTPFile(io.RawIOBase):
    """Random-access file-like object backed by HTTP Range requests."""

    def __init__(self, url, session):
        self.url, self.s = url, session
        self.pos = self.fetched = 0
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Range"].split("/")[1])

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n <= 0:
            return b""
        end = min(self.pos + n, self.size) - 1
        r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=300)
        r.raise_for_status()
        d = r.content
        self.pos += len(d)
        self.fetched += len(d)
        return d


def list_participants(session):
    html = session.get(BASE, timeout=60).text
    ids = sorted({int(m) for m in re.findall(r'href="(\d+)_P\.zip"', html)})
    return ids


def fetch_one(pid, out, wanted, session):
    """Returns (pid, bytes_fetched, [written paths]) or raises."""
    url = f"{BASE}{pid}_P.zip"
    written, targets = [], [SELECTORS[w] for w in wanted]

    # skip if everything already on disk
    todo = []
    for w in wanted:
        sub = SUBDIR.get(w, w)
        # filename inside the zip is <pid><suffix>
        dest = os.path.join(out, sub, f"{pid}{SELECTORS[w]}")
        if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
            todo.append((w, dest))
    if not todo:
        return pid, 0, ["cached"]

    f = HTTPFile(url, session)
    z = zipfile.ZipFile(f)
    names = z.namelist()
    for w, dest in todo:
        suffix = SELECTORS[w]
        # Some archives were built on macOS and carry an AppleDouble sidecar
        # (._<name>) alongside the real member -- match the basename exactly so
        # we never pick up the 4KB resource fork instead of the transcript.
        match = [n for n in names
                 if os.path.basename(n) == f"{pid}{suffix}"
                 and "__MACOSX" not in n]
        if not match:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(z.read(match[0]))
        os.replace(tmp, dest)
        written.append(dest)
    return pid, f.fetched, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", default="audio,transcript",
                    help="comma list from: " + ",".join(SELECTORS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only first N participants (smoke test)")
    args = ap.parse_args()

    wanted = [m.strip() for m in args.members.split(",") if m.strip()]
    bad = [w for w in wanted if w not in SELECTORS]
    if bad:
        sys.exit(f"unknown members: {bad}")

    os.makedirs(args.out, exist_ok=True)
    session = requests.Session()
    # The host sits behind a WAF that 403s both the default requests UA and a
    # spoofed browser UA on the directory index; a curl UA is what it accepts.
    session.headers["User-Agent"] = "curl/7.81.0"

    # metadata / label files
    lab = os.path.join(args.out, "labels")
    os.makedirs(lab, exist_ok=True)
    for m in META:
        dest = os.path.join(lab, m)
        if not os.path.exists(dest):
            r = session.get(BASE + m, timeout=120)
            if r.ok:
                open(dest, "wb").write(r.content)
    print(f"[meta] {len(os.listdir(lab))} files in {lab}", flush=True)

    pids = list_participants(session)
    if args.limit:
        pids = pids[: args.limit]
    print(f"[plan] {len(pids)} participants, members={wanted}, workers={args.workers}", flush=True)

    t0, total, done, errs = time.time(), 0, 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, p, args.out, wanted, session): p for p in pids}
        for fu in as_completed(futs):
            pid = futs[fu]
            try:
                pid, nb, w = fu.result()
                total += nb
                done += 1
                if done % 20 == 0 or done == len(pids):
                    el = time.time() - t0
                    print(f"[{done}/{len(pids)}] {total/1e9:.2f} GB fetched, "
                          f"{el:.0f}s, {total/1e6/max(el,1):.0f} MB/s", flush=True)
            except Exception as e:
                errs.append((pid, repr(e)))
                print(f"[ERR] {pid}: {e!r}", flush=True)

    el = time.time() - t0
    print(f"\n[done] {done}/{len(pids)} ok, {len(errs)} errors, "
          f"{total/1e9:.2f} GB transferred in {el:.0f}s", flush=True)
    if errs:
        print("[failed]", errs, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
