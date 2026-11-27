# -*- coding: utf-8 -*-
"""分块 Range 下载（绕开 hf-mirror 长连接 SSL 断流）。

用法:
  python download_range.py --url <url> --out <path> [--chunk 8388608] [--retries 12]
"""
import argparse
import os
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=8 * 1024 * 1024)
    ap.add_argument("--retries", type=int, default=12)
    args = ap.parse_args()

    expected = None
    try:
        h = requests.head(args.url, allow_redirects=True, timeout=30,
                          headers={"Accept-Encoding": "identity"})
        cl = h.headers.get("content-length")
        if cl:
            expected = int(cl)
        print(f"HEAD {h.status_code} size={expected}", flush=True)
    except Exception as e:
        print(f"HEAD failed: {e}", flush=True)

    if os.path.exists(args.out) and expected and os.path.getsize(args.out) == expected:
        print(f"already complete {args.out} size={expected}", flush=True)
        return

    if os.path.exists(args.out):
        os.remove(args.out)

    pos = 0
    with open(args.out, "wb") as f:
        while True:
            end = pos + args.chunk - 1
            headers = {"Range": f"bytes={pos}-{end}", "Accept-Encoding": "identity"}
            data = None
            for attempt in range(args.retries):
                try:
                    r = requests.get(args.url, headers=headers, stream=True, timeout=90)
                    if r.status_code == 416:
                        data = b""
                        break
                    if r.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {r.status_code}")
                    data = r.content
                    r.close()
                    if data is None:
                        raise RuntimeError("empty")
                    break
                except Exception as e:
                    print(f"retry {attempt} at {pos}: {e}", flush=True)
                    time.sleep(2 + attempt)
            if data is None:
                raise SystemExit(f"failed at byte {pos}")
            if not data:
                break
            f.write(data)
            pos += len(data)
            pct = f"{pos}/{expected}" if expected else f"{pos}"
            print(pct, flush=True)
            if len(data) < args.chunk:
                break

    size = os.path.getsize(args.out)
    if expected and size != expected:
        raise SystemExit(f"size mismatch: {size} != {expected}")
    print(f"done {args.out} size={size}", flush=True)


if __name__ == "__main__":
    main()
