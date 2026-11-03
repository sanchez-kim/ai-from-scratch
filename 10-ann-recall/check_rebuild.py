"""Rebuild one index from scratch and compare its sha256 with the earlier builds.

Builds data/<set>-c.bin in this process (1 thread, same settings as
build_index.py), records the start and end time, and writes the sha256 of the
a, b and c files to results/rebuild-<set>.json.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import time

import numpy as np

from build_index import EF_CONSTRUCTION, M, SEED, build

HERE = pathlib.Path(__file__).parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["real", "random"], required=True)
    args = ap.parse_args()
    out_path = HERE / "data" / f"{args.set}-c.bin"
    if out_path.exists():
        out_path.unlink()  # never compare against a stale or copied file
    corpus = np.load(HERE / "data" / f"{args.set}.npz")["corpus"]
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    build(corpus).save_index(str(out_path))
    finished = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    files = {}
    for tag in "abc":
        p = HERE / "data" / f"{args.set}-{tag}.bin"
        files[p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size,
                         "mtime": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(os.stat(p).st_mtime))}
    out = {"uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
           "set": args.set, "M": M, "ef_construction": EF_CONSTRUCTION, "random_seed": SEED, "build_threads": 1,
           "c_build_started": started, "c_build_finished": finished, "pid": os.getpid(), "files": files,
           "all_three_identical": len({f["sha256"] for f in files.values()}) == 1}
    (HERE / "results" / f"rebuild-{args.set}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
