# -*- coding: utf-8 -*-
"""Split a large safetensors file into smaller shards without loading tensors.

Raw byte copy: parse the header, partition tensor entries into groups under a
target size, write each group as a valid safetensors file, and rewrite the
model.safetensors.index.json weight_map.

Usage:
  python split_safetensors.py --src <dir>/model-00001-of-00002.safetensors \
      --parts 3 --target-gb 3.2 --dir <dir>
"""
import argparse
import json
import os
import struct


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return n, header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--parts", type=int, default=3)
    ap.add_argument("--target-gb", type=float, default=3.2)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prefix", default="model-split")
    args = ap.parse_args()

    n, header = read_header(args.src)
    meta = header.pop("__metadata__", None)
    items = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    total = sum(v["data_offsets"][1] - v["data_offsets"][0] for _, v in items)
    print(f"src={args.src} header={n}B tensors={len(items)} data={total/1e9:.2f}GB", flush=True)

    target = int(args.target_gb * 1e9)
    groups, cur, cur_sz = [], [], 0
    for name, v in items:
        sz = v["data_offsets"][1] - v["data_offsets"][0]
        if cur and cur_sz + sz > target:
            groups.append(cur)
            cur, cur_sz = [], 0
        cur.append((name, v))
        cur_sz += sz
    if cur:
        groups.append(cur)
    print(f"groups={len(groups)} sizes={[round(sum(v['data_offsets'][1]-v['data_offsets'][0] for _,v in g)/1e9,2) for g in groups]}", flush=True)

    out_names = []
    with open(args.src, "rb") as src:
        for gi, g in enumerate(groups):
            out = os.path.join(args.dir, f"{args.prefix}-{gi+1:05d}.safetensors")
            new_header = {}
            if meta:
                new_header["__metadata__"] = meta
            off = 0
            for name, v in g:
                sz = v["data_offsets"][1] - v["data_offsets"][0]
                new_header[name] = {"dtype": v["dtype"], "shape": v["shape"],
                                    "data_offsets": [off, off + sz]}
                off += sz
            hjson = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
            pad = (-len(hjson)) % 8
            hjson += b" " * pad
            with open(out, "wb") as w:
                w.write(struct.pack("<Q", len(hjson)))
                w.write(hjson)
                for name, v in g:
                    s, e = v["data_offsets"]
                    src.seek(8 + n + s)
                    left = e - s
                    while left > 0:
                        chunk = src.read(min(left, 64 << 20))
                        if not chunk:
                            raise RuntimeError("unexpected EOF")
                        w.write(chunk)
                        left -= len(chunk)
            print(f"wrote {out} {os.path.getsize(out)/1e9:.2f}GB tensors={len(g)}", flush=True)
            out_names.append(os.path.basename(out))

    idx_path = os.path.join(args.dir, "model.safetensors.index.json")
    with open(idx_path, encoding="utf-8") as f:
        idx = json.load(f)
    wm = idx.get("weight_map", {})
    # map every tensor that lived in src to its new file
    src_name = os.path.basename(args.src)
    newmap = {}
    for gi, g in enumerate(groups):
        for name, _ in g:
            newmap[name] = out_names[gi]
    repl = 0
    for k, v in list(wm.items()):
        if v == src_name:
            wm[k] = newmap[k]
            repl += 1
    idx["weight_map"] = wm
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)
    print(f"index updated, replaced {repl} entries", flush=True)


if __name__ == "__main__":
    main()
