#!/usr/bin/env python3
"""Cross-check watermark.py (Python) against watermark-rs (Rust).

    python crosstest.py --py path/to/watermark.py --rs target/release/watermark-rs

Checks, in order:
  1. embed with Python  -> extract with Rust
  2. embed with Rust    -> extract with Python
  3. Rust extract on processed copies (resize, JPEG, small rotation), compared with Python
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

MESSAGE = "Short text"


def make_image(path: Path, w: int = 1000, h: int = 800) -> None:
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 128 + 40 * np.sin(xx * 0.013) * np.cos(yy * 0.011)
    noise = rng.normal(0, 12, (h, w))
    gray = np.clip(base + noise, 0, 255)
    rgb = np.stack([gray, np.clip(gray + 8, 0, 255), np.clip(gray - 8, 0, 255)], axis=-1)
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def rs(binary: str, *args: str) -> tuple[int, str]:
    p = subprocess.run([binary, *args], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def py(script: str, *args: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, script, *args], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", required=True, help="path to watermark.py")
    ap.add_argument("--rs", required=True, help="path to the watermark-rs binary")
    args = ap.parse_args()

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src.png"
        make_image(src)

        # 1. Python embed -> Rust extract
        py_marked = tmp / "py_marked.png"
        code, out = py(args.py, "embed", str(src), str(py_marked), MESSAGE)
        results.append(report("python embed", code == 0, out))
        code, out = rs(args.rs, "extract", str(py_marked))
        results.append(report("py embed -> rust extract", code == 0 and out == MESSAGE, out))

        # 2. Rust embed -> Python extract
        rs_marked = tmp / "rs_marked.png"
        code, out = rs(args.rs, "embed", str(src), str(rs_marked), MESSAGE)
        results.append(report("rust embed", code == 0, out))
        code, out = py(args.py, "extract", str(rs_marked))
        results.append(report("rust embed -> py extract", code == 0 and out == MESSAGE, out))

        # 3. Processed copies of the Rust-marked image
        im = Image.open(rs_marked).convert("RGB")
        variants = {
            "resize x1.25": im.resize((round(im.width * 1.25), round(im.height * 1.25)), Image.BICUBIC),
            "resize x0.8": im.resize((round(im.width * 0.8), round(im.height * 0.8)), Image.LANCZOS),
            "rotate 0.5 deg": im.rotate(0.5, resample=Image.BICUBIC, fillcolor=(128, 128, 128)),
        }
        for name, img in variants.items():
            path = tmp / f"{name.replace(' ', '_')}.png"
            img.save(path)
            code, out = rs(args.rs, "extract", str(path))
            results.append(report(f"rust extract: {name}", code == 0 and out == MESSAGE, out))
        jpg = tmp / "recompressed.jpg"
        im.save(jpg, quality=70)
        code, out = rs(args.rs, "extract", str(jpg))
        results.append(report("rust extract: JPEG q70", code == 0 and out == MESSAGE, out))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())