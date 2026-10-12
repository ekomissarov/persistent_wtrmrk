#!/usr/bin/env python3
"""Document-aware public blind watermark (experimental print/scan prototype).

Usage:
  python watermark.py embed photo.jpg marked.png "Short text"
  python watermark.py embed photo.jpg marked.png "Short text" --debug
  python watermark.py extract marked.png

One fixed algorithm: adaptive DCT pair modulation, repeated 320x320-pixel tiles,
public synchronization pilots, interleaved Hamming(15,11), CRC32.
No secrets, signatures, or proof of image authorship. In particular anyone
who knows this algorithm can overwrite or intentionally disrupt the watermark.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn, idctn

BLOCK = 16
TILE = 20
MAX_MESSAGE_BYTES = 16
FRAME_BYTES = 1 + MAX_MESSAGE_BYTES + 4  # length, padded text, CRC32
FRAME_BITS = FRAME_BYTES * 8
PARITY_BITS = 4
CODE_BITS = 15 * math.ceil(FRAME_BITS / 11)  # Hamming(15, 11): 240 bits
STRENGTH = 115.0  # maximum DCT margin on textured non-text regions
MAX_PIXELS = 16_000_000
COEFF_A = (2, 3)
COEFF_B = (3, 2)
SEED = 0x57A73  # public fixed pattern, not a secret key

_rng = np.random.default_rng(SEED)
_permutation = _rng.permutation(TILE * TILE)
PILOTS = TILE * TILE - CODE_BITS
pilot_indices = _permutation[:PILOTS]
data_indices = _permutation[PILOTS:]
pilot_template = np.zeros(TILE * TILE, dtype=np.float32)
pilot_template[pilot_indices] = _rng.choice([-1.0, 1.0], PILOTS)
pilot_template = pilot_template.reshape(TILE, TILE)
mask_bits = _rng.integers(0, 2, size=CODE_BITS, dtype=np.uint8)
pilot_fft = np.conj(np.fft.fft2(pilot_template))
pilot_abs_fft = np.conj(np.fft.fft2(np.abs(pilot_template)))
HAMMING_DATA = [i for i in range(1, 16) if i not in (1, 2, 4, 8)]


def _frame(message: str) -> np.ndarray:
    text = message.encode('utf-8')
    if not (1 <= len(text) <= MAX_MESSAGE_BYTES):
        raise ValueError('Message must contain 1–16 UTF-8 bytes (not characters).')
    header = bytes([len(text)]) + text.ljust(MAX_MESSAGE_BYTES, b'\0')
    return np.unpackbits(np.frombuffer(header + (zlib.crc32(header) & 0xffffffff).to_bytes(4, 'big'), dtype=np.uint8))


def _unframe(bits: np.ndarray) -> str | None:
    raw = np.packbits(bits).tobytes()[:FRAME_BYTES]
    header, check = raw[:-4], raw[-4:]
    if (zlib.crc32(header) & 0xffffffff).to_bytes(4, 'big') != check:
        return None
    length = header[0]
    if not (1 <= length <= MAX_MESSAGE_BYTES) or any(header[1+length:]):
        return None
    try:
        return header[1:1+length].decode('utf-8')
    except UnicodeDecodeError:
        return None


def _encode_hamming(bits: np.ndarray) -> np.ndarray:
    padded = np.pad(bits, (0, -len(bits) % 11)).reshape(-1, 11)
    encoded = np.zeros((len(padded), 16), dtype=np.uint8)
    encoded[:, HAMMING_DATA] = padded
    for p in (1, 2, 4, 8):
        encoded[:, p] = np.bitwise_xor.reduce(encoded[:, [x for x in range(1, 16) if x & p]], axis=1)
    return encoded[:, 1:].ravel()


def _decode_hamming(bits: np.ndarray) -> np.ndarray:
    words = np.zeros((len(bits) // 15, 16), dtype=np.uint8)
    words[:, 1:] = bits.reshape(-1, 15)
    syndrome = np.zeros(len(words), dtype=np.uint8)
    for p in (1, 2, 4, 8):
        parity = np.bitwise_xor.reduce(words[:, [x for x in range(1, 16) if x & p]], axis=1)
        syndrome += p * parity
    bad = np.nonzero(syndrome)[0]
    words[bad, syndrome[bad]] ^= 1
    return words[:, HAMMING_DATA].ravel()[:FRAME_BITS]


def _pattern(message: str) -> np.ndarray:
    signs = pilot_template.ravel().copy()
    encoded = _encode_hamming(_frame(message)) ^ mask_bits
    signs[data_indices] = np.where(encoded != 0, 1.0, -1.0)
    return signs.reshape(TILE, TILE)


def _load(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.width * im.height > MAX_PIXELS:
            raise ValueError(f'Image exceeds {MAX_PIXELS:,} pixels.')
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB'), dtype=np.uint8).copy()


def _save(rgb: np.ndarray, output: Path) -> None:
    ext = output.suffix.lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        raise ValueError('Output must be a PNG or JPEG file.')
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb)
    opts = {'quality': 95, 'subsampling': 0} if ext in ('.jpg', '.jpeg') else {}
    fmt = 'JPEG' if ext in ('.jpg', '.jpeg') else 'PNG'
    # Safe no-clobber publishing. Never remove a file belonging to another writer.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.watermark-', suffix='.tmp', delete=False) as fh:
            tmp = Path(fh.name)
            image.save(fh, format=fmt, **opts)
        os.link(tmp, output)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def embed(rgb: np.ndarray, message: str, debug: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Embed in textured regions while sparing paper, characters and flat areas.

    The mask is deterministic from local image statistics, so extraction does
    not need the original. This is *not* OCR or a certified document mask.
    """
    h, w = rgb.shape[:2]
    if h < BLOCK * TILE or w < BLOCK * TILE:
        raise ValueError(f'Image must be at least {BLOCK * TILE}x{BLOCK * TILE} pixels.')
    pattern = _pattern(message)
    bh, bw = h // BLOCK, w // BLOCK
    hh, ww = bh * BLOCK, bw * BLOCK
    pixels = rgb.astype(np.float32)
    y = .299 * pixels[:, :, 0] + .587 * pixels[:, :, 1] + .114 * pixels[:, :, 2]
    blocks = y[:hh, :ww].reshape(bh, BLOCK, bw, BLOCK).transpose(0, 2, 1, 3).copy()

    # Compute image-aware embedding mask using 16x16 block statistics.
    # White paper is skipped; fine high-contrast edges (often glyph strokes)
    # are skipped; continuous midtone/texture is preferred.
    mean = blocks.mean(axis=(2, 3))
    std = blocks.std(axis=(2, 3))
    bright_paper = np.clip((246.0 - mean) / 32.0, 0, 1)
    textured = np.clip((std - 5.0) / 20.0, 0, 1)
    # Detect likely black-on-white text (very bright pixels + very dark pixels
    # in the same tile). Avoid modifying these tiles.
    white_fraction = (blocks >= 235).mean(axis=(2, 3))
    black_fraction = (blocks <= 100).mean(axis=(2, 3))
    text_like = (white_fraction > .40) & (black_fraction > .015)
    weight = bright_paper * textured * (~text_like)
    # Absolute minimum mask: no pixel modifications on excluded blocks.
    active = weight >= .20

    c = dctn(blocks, axes=(-2, -1), norm='ortho')
    a = c[:, :, COEFF_A[0], COEFF_A[1]]
    b = c[:, :, COEFF_B[0], COEFF_B[1]]
    wanted = np.tile(pattern, (math.ceil(bh / TILE), math.ceil(bw / TILE)))[:bh, :bw]
    target = STRENGTH * weight
    correction = np.where(active, np.maximum(0., target - (a - b) * wanted), 0.) * wanted / 2
    modified = active & (np.abs(correction) > 1e-6)
    a += correction
    b -= correction
    updated_y = idctn(c, axes=(-2, -1), norm='ortho').transpose(0, 2, 1, 3).reshape(hh, ww)
    pixels[:hh, :ww] += (updated_y - y[:hh, :ww])[:, :, None]
    result = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
    if not debug:
        return result
    debug_image = result.copy()
    for by, bx in np.argwhere(modified):
        x1, y1 = int(bx * BLOCK), int(by * BLOCK)
        cv2.rectangle(debug_image, (x1, y1), (x1 + BLOCK - 1, y1 + BLOCK - 1), (255, 0, 0), 1)
    return result, debug_image


def _basis(index: int) -> np.ndarray:
    x = np.arange(BLOCK, dtype=np.float32)
    return (math.sqrt(2 / BLOCK) * np.cos(math.pi * index * (x + .5) / BLOCK)).astype(np.float32)


def _differences(y: np.ndarray) -> np.ndarray:
    a = cv2.sepFilter2D(y, cv2.CV_32F, _basis(3), _basis(2), anchor=(8, 8))
    b = cv2.sepFilter2D(y, cv2.CV_32F, _basis(2), _basis(3), anchor=(8, 8))
    return a - b


def _fold(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = data.shape
    ph, pw = -h % TILE, -w % TILE
    padded = np.pad(data, ((0, ph), (0, pw)))
    sums = padded.reshape(-1, TILE, padded.shape[1] // TILE, TILE).sum(axis=(0, 2), dtype=np.float32)
    cy = np.bincount(np.arange(h) % TILE, minlength=TILE).astype(np.float32)
    cx = np.bincount(np.arange(w) % TILE, minlength=TILE).astype(np.float32)
    return sums, np.outer(cy, cx)


def _decode(sums: np.ndarray, counts: np.ndarray, tile_y: int, tile_x: int, deep: bool = False) -> str | None:
    sums = np.roll(sums, (-tile_y, -tile_x), axis=(0, 1)).ravel()
    counts = np.roll(counts, (-tile_y, -tile_x), axis=(0, 1)).ravel()
    soft = sums[data_indices] / np.maximum(1., counts[data_indices])
    bits = (soft > 0).astype(np.uint8) ^ mask_bits
    result = _unframe(_decode_hamming(bits))
    if result is not None:
        return result
    # Try alternative decisions at the least reliable bit positions.
    uncertain = np.argsort(np.abs(soft))[:16 if deep else 8]
    for i in uncertain:
        alt = bits.copy()
        alt[i] ^= 1
        result = _unframe(_decode_hamming(alt))
        if result is not None:
            return result
    if deep:
        # Two-bit ambiguity recovery (CRC32 validates the whole message).
        for x, i in enumerate(uncertain[:12]):
            for j in uncertain[x+1:12]:
                alt = bits.copy()
                alt[i] ^= 1
                alt[j] ^= 1
                result = _unframe(_decode_hamming(alt))
                if result is not None:
                    return result
    return None


def _detect(y: np.ndarray, deep: bool = False) -> tuple[str | None, float]:
    """Search DCT block alignment and periodic pilot phase."""
    if min(y.shape) < 130:
        return None, 0.
    diff = _differences(y)
    h, w = diff.shape
    candidates = []
    phases = 10 if deep else 2
    for oy in range(BLOCK):
        rows = np.arange(8 + oy, h - 8, BLOCK)
        if len(rows) < 2:
            continue
        for ox in range(BLOCK):
            cols = np.arange(8 + ox, w - 8, BLOCK)
            if len(cols) < 2:
                continue
            sampled = diff[np.ix_(rows, cols)]
            sums, counts = _fold(sampled)
            correlation = np.fft.ifft2(np.fft.fft2(sums) * pilot_fft).real
            observations = np.fft.ifft2(np.fft.fft2(counts) * pilot_abs_fft).real
            z = correlation / ((float(np.std(sampled)) + 1e-4) * np.sqrt(np.maximum(1, observations)))
            flat = z.ravel()
            for idx in np.argpartition(flat, -phases)[-phases:]:
                if deep or flat[idx] > 2.5:
                    sy, sx = np.unravel_index(idx, z.shape)
                    candidates.append((float(flat[idx]), sums, counts, sy, sx))
    candidates.sort(key=lambda c: c[0], reverse=True)
    for score, sums, counts, sy, sx in candidates[:120 if deep else 40]:
        result = _decode(sums, counts, sy, sx, deep=deep)
        if result is not None:
            return result, score
    return None, candidates[0][0] if candidates else 0.


def _rotate(y: np.ndarray, angle: float) -> np.ndarray:
    h, w = y.shape
    centre = ((w - 1) / 2., (h - 1) / 2.)
    m = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return cv2.warpAffine(y, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def extract(rgb: np.ndarray) -> str | None:
    """Fast search, then a more exhaustive search without changing the watermark."""
    if rgb.shape[0] * rgb.shape[1] > MAX_PIXELS:
        raise ValueError(f'Image exceeds {MAX_PIXELS:,} pixels.')
    a = rgb.astype(np.float32)
    y = .299*a[:, :, 0] + .587*a[:, :, 1] + .114*a[:, :, 2]
    common = (1., 2., 1.333333333, 1.25, 1.428571429, 1.666666667,
              1.176470588, 1.111111111, 1.5, .8, .666666667, .5, 1.2)
    angles = (0., -.5, .5, -1., 1.)
    ranked = []

    def check(upright: np.ndarray, multiplier: float, deep: bool = False, collect: bool = False) -> str | None:
        hh, ww = upright.shape
        width, height = round(ww * multiplier), round(hh * multiplier)
        if min(height, width) < 128 or (multiplier > 1 and max(height, width) > 3600) or height * width > MAX_PIXELS:
            return None
        scaled = (upright if multiplier == 1 else
                  cv2.resize(upright, (width, height), interpolation=cv2.INTER_CUBIC if multiplier > 1 else cv2.INTER_AREA))
        found, strength = _detect(scaled, deep=deep)
        if collect:
            ranked.append((strength, multiplier))
        return found

    # Fast path: common resize ratios and minor deskew angles.
    for angle in angles:
        upright = y if angle == 0 else _rotate(y, angle)
        for scale in common:
            result = check(upright, scale)
            if result is not None:
                return result

    # Deep path: fewer early rejections, more phases, bit decisions and candidates.
    for angle in angles:
        upright = y if angle == 0 else _rotate(y, angle)
        for scale in common:
            result = check(upright, scale, deep=True)
            if result is not None:
                return result

    # Fine scale search without rotation first; collect promising scale hypotheses.
    fine = sorted(np.arange(.75, 1.551, .005), key=lambda s: abs(math.log(s)))
    for scale in fine:
        if min(abs(scale - c) for c in common) < .003:
            continue
        result = check(y, float(scale), collect=True)
        if result is not None:
            return result

    selected = []
    for _, scale in sorted(ranked, reverse=True):
        if all(abs(scale - old) >= .02 for old in selected):
            selected.append(float(scale))
        if len(selected) == 8:
            break

    # Spend the additional time on the most promising scales and nearby rotations.
    for scale in selected:
        result = check(y, scale, deep=True)
        if result is not None:
            return result
    for angle in angles[1:]:
        upright = _rotate(y, angle)
        for scale in selected:
            for offset in (-.01, 0., .01):
                result = check(upright, scale + offset, deep=True)
                if result is not None:
                    return result
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description='Invisible image watermark (experimental print/scan resistance)')
    commands = parser.add_subparsers(dest='command', required=True)
    encoder = commands.add_parser('embed', help='Embed up to 16 UTF-8 bytes')
    encoder.add_argument('input', type=Path)
    encoder.add_argument('output', type=Path)
    encoder.add_argument('message')
    encoder.add_argument('--debug', action='store_true', help='Save a PNG with red borders around modified DCT blocks')
    decoder = commands.add_parser('extract', help='Read the hidden text; no original required')
    decoder.add_argument('input', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'embed':
            if args.output.exists():
                raise FileExistsError(f'Output already exists: {args.output}')
            _frame(args.message)
            debug_path = args.output.with_name(args.output.stem + '_debug.png') if args.debug else None
            if debug_path is not None and debug_path.exists():
                raise FileExistsError(f'Debug output already exists: {debug_path}')
            result = embed(_load(args.input), args.message, debug=args.debug)
            image, debug_image = result if args.debug else (result, None)
            _save(image, args.output)
            print(f'Watermark embedded: {args.output}')
            if args.debug:
                _save(debug_image, debug_path)
                print(f'Debug visualization: {debug_path}')
        else:
            result = extract(_load(args.input))
            if result is None:
                print('Watermark not found.', file=sys.stderr)
                return 1
            print(result)
        return 0
    except (ValueError, OSError, cv2.error, MemoryError, Image.DecompressionBombError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
