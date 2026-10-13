#!/usr/bin/env python3
"""Document-aware public blind watermark (experimental print/scan prototype).

Usage:
  python watermark.py embed photo.jpg marked.png "Short text"
  python watermark.py embed photo.jpg marked.png "Short text" --debug
  python watermark.py extract marked.png

One fixed algorithm: adaptive DCT pair modulation, repeated 320x320-pixel tiles,
public synchronization pilots, Hamming(15,11), 48-bit checksum.
No secrets, signatures, or proof of image authorship. In particular anyone
who knows this algorithm can overwrite or intentionally disrupt the watermark.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn, idctn

# --------------------------------------------------------------------------- #
# Format constants (changing any of these breaks compatibility with marked files)
# --------------------------------------------------------------------------- #
BLOCK = 16                      # DCT block size, pixels
TILE = 20                       # repeated pattern size, blocks (320 px)
MAX_MESSAGE_BYTES = 16
CHECKSUM_BYTES = 6              # truncated BLAKE2s
FRAME_BYTES = 1 + MAX_MESSAGE_BYTES + CHECKSUM_BYTES      # length + text + checksum
FRAME_BITS = FRAME_BYTES * 8                              # 184
CODE_BITS = 15 * math.ceil(FRAME_BITS / 11)               # 17 codewords -> 255
PILOTS = TILE * TILE - CODE_BITS                          # 145
COEFF_A = (2, 3)
COEFF_B = (3, 2)
STRENGTH = 115.0                # maximum DCT margin on textured non-text regions
SEED = 0x57A73                  # public fixed pattern, not a secret key

# --------------------------------------------------------------------------- #
# Operational limits (free to tune; they do not affect the watermark format)
# --------------------------------------------------------------------------- #
MIN_EMBED_SIDE = BLOCK * TILE
MIN_DETECT_SIDE = 130
MAX_PIXELS = 50_000_000
MAX_EXTRACT_SIDE = 3600
SEARCH_SECONDS = 110
EMBED_ACTIVE_WEIGHT = 0.20      # blocks lighter than this are not modified
DETECT_MIN_WEIGHT = 0.15        # slightly looser at detection (content shifts after embedding)
PROGRESS_INTERVAL = 12          # seconds between progress messages
JPEG_OPTIONS = {'quality': 95, 'subsampling': 0}
OUTPUT_FORMATS = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG'}


# --------------------------------------------------------------------------- #
# Deterministic public layout: data/pilot positions, pilot signs, payload mask
# --------------------------------------------------------------------------- #
def _random_bytes(n: int, domain: bytes) -> bytes:
    """Version-independent deterministic stream; no secret key."""
    seed = SEED.to_bytes(8, 'big')
    out = bytearray()
    for counter in range((n + 31) // 32):
        out.extend(hashlib.sha256(domain + seed + counter.to_bytes(8, 'big')).digest())
    return bytes(out[:n])


def _random_bits(n: int, domain: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(_random_bytes(math.ceil(n / 8), domain), dtype=np.uint8))[:n]


def _shuffle(n: int) -> np.ndarray:
    """Fisher-Yates with fixed SHA256-derived words, independent of NumPy's RNG."""
    indices = list(range(n))
    source = _random_bytes(4 * n, b'layout-v5')
    for i in range(n - 1, 0, -1):
        j = int.from_bytes(source[4 * i:4 * i + 4], 'big') % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    return np.array(indices, dtype=np.int32)


_PERMUTATION = _shuffle(TILE * TILE)
PILOT_INDICES = _PERMUTATION[:PILOTS]
DATA_INDICES = _PERMUTATION[PILOTS:]

PILOT_TEMPLATE = np.zeros(TILE * TILE, dtype=np.float32)
PILOT_TEMPLATE[PILOT_INDICES] = np.where(_random_bits(PILOTS, b'pilots-v5') != 0, 1., -1.)
PILOT_TEMPLATE = PILOT_TEMPLATE.reshape(TILE, TILE)
MASK_BITS = _random_bits(CODE_BITS, b'mask-v5').copy()

PILOT_FFT = np.conj(np.fft.fft2(PILOT_TEMPLATE))
PILOT_ABS_FFT = np.conj(np.fft.fft2(np.abs(PILOT_TEMPLATE)))


# --------------------------------------------------------------------------- #
# Message framing and Hamming(15,11)
# --------------------------------------------------------------------------- #
_HAMMING_PARITY_POSITIONS = (1, 2, 4, 8)
_HAMMING_DATA = [i for i in range(1, 16) if i not in _HAMMING_PARITY_POSITIONS]
_HAMMING_COVERS = {p: [x for x in range(1, 16) if x & p] for p in _HAMMING_PARITY_POSITIONS}


def _checksum(header: bytes) -> bytes:
    return hashlib.blake2s(header, digest_size=CHECKSUM_BYTES).digest()


def _frame(message: str) -> np.ndarray:
    text = message.encode('utf-8')
    if not 1 <= len(text) <= MAX_MESSAGE_BYTES:
        raise ValueError(f'Message must contain 1–{MAX_MESSAGE_BYTES} UTF-8 bytes (not characters).')
    header = bytes([len(text)]) + text.ljust(MAX_MESSAGE_BYTES, b'\0')
    return np.unpackbits(np.frombuffer(header + _checksum(header), dtype=np.uint8))


def _unframe(bits: np.ndarray) -> str | None:
    raw = np.packbits(bits).tobytes()[:FRAME_BYTES]
    header, check = raw[:-CHECKSUM_BYTES], raw[-CHECKSUM_BYTES:]
    if _checksum(header) != check:
        return None
    length = header[0]
    if not 1 <= length <= MAX_MESSAGE_BYTES or any(header[1 + length:]):
        return None
    try:
        return header[1:1 + length].decode('utf-8')
    except UnicodeDecodeError:
        return None


def _encode_hamming(bits: np.ndarray) -> np.ndarray:
    padded = np.pad(bits, (0, -len(bits) % 11)).reshape(-1, 11)
    words = np.zeros((len(padded), 16), dtype=np.uint8)
    words[:, _HAMMING_DATA] = padded
    for p, covered in _HAMMING_COVERS.items():
        words[:, p] = np.bitwise_xor.reduce(words[:, covered], axis=1)
    return words[:, 1:].ravel()


def _decode_hamming(bits: np.ndarray) -> np.ndarray:
    words = np.zeros((len(bits) // 15, 16), dtype=np.uint8)
    words[:, 1:] = bits.reshape(-1, 15)
    syndrome = np.zeros(len(words), dtype=np.uint8)
    for p, covered in _HAMMING_COVERS.items():
        syndrome += p * np.bitwise_xor.reduce(words[:, covered], axis=1)
    bad = np.nonzero(syndrome)[0]
    words[bad, syndrome[bad]] ^= 1
    return words[:, _HAMMING_DATA].ravel()[:FRAME_BITS]


def _decode_bits(bits: np.ndarray) -> str | None:
    return _unframe(_decode_hamming(bits))


def _pattern(message: str) -> np.ndarray:
    """+/-1 sign for each block of one tile: pilots plus masked, coded payload."""
    signs = PILOT_TEMPLATE.ravel().copy()
    coded = _encode_hamming(_frame(message)) ^ MASK_BITS
    signs[DATA_INDICES] = np.where(coded != 0, 1., -1.)
    return signs.reshape(TILE, TILE)


# --------------------------------------------------------------------------- #
# Image I/O
# --------------------------------------------------------------------------- #
def _load(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.width * im.height > MAX_PIXELS:
            raise ValueError(f'Image exceeds {MAX_PIXELS:,} pixels.')
        im = ImageOps.exif_transpose(im)
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            # Flatten transparency onto white instead of letting convert() keep stray colours.
            rgba = im.convert('RGBA')
            background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(background, rgba)
        return np.asarray(im.convert('RGB'), dtype=np.uint8).copy()


def _output_format(path: Path) -> str:
    try:
        return OUTPUT_FORMATS[path.suffix.lower()]
    except KeyError:
        raise ValueError('Output must be a PNG or JPEG file.') from None


def _encode_image(rgb: np.ndarray, fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format=fmt, **(JPEG_OPTIONS if fmt == 'JPEG' else {}))
    return buffer.getvalue()


def _write_new(data: bytes, output: Path) -> None:
    """Write `data` to a file that must not exist yet; remove it again on failure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output.open('xb') as fh:   # exclusive creation also works without hard links
            created = True
            fh.write(data)
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise


def _luma(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[:, :, i].astype(np.float32) for i in range(3))
    return .299 * r + .587 * g + .114 * b


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def _embedding_weight(blocks: np.ndarray) -> np.ndarray:
    """Per-block weight in [0, 1]: textured, non-white, non-text regions score high."""
    mean = blocks.mean(axis=(2, 3))
    std = blocks.std(axis=(2, 3))
    not_paper = np.clip((246.0 - mean) / 32.0, 0, 1)
    textured = np.clip((std - 5.0) / 20.0, 0, 1)
    white_fraction = (blocks >= 235).mean(axis=(2, 3))
    black_fraction = (blocks <= 100).mean(axis=(2, 3))
    text_like = (white_fraction > .40) & (black_fraction > .015)
    return not_paper * textured * (~text_like)


def embed(rgb: np.ndarray, message: str, debug: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Embed in textured regions while sparing paper, characters and flat areas.

    The mask is deterministic from local image statistics, so extraction does
    not need the original. This is *not* OCR or a certified document mask.
    """
    h, w = rgb.shape[:2]
    if h < MIN_EMBED_SIDE or w < MIN_EMBED_SIDE:
        raise ValueError(f'Image must be at least {MIN_EMBED_SIDE}x{MIN_EMBED_SIDE} pixels.')
    pattern = _pattern(message)
    bh, bw = h // BLOCK, w // BLOCK
    hh, ww = bh * BLOCK, bw * BLOCK
    pixels = rgb.astype(np.float32)
    y = _luma(rgb)[:hh, :ww]
    blocks = y.reshape(bh, BLOCK, bw, BLOCK).transpose(0, 2, 1, 3).copy()

    weight = _embedding_weight(blocks)
    active = weight >= EMBED_ACTIVE_WEIGHT
    if not active.any():
        raise ValueError('No suitable blocks: watermark was not embedded.')

    coeffs = dctn(blocks, axes=(-2, -1), norm='ortho')
    a = coeffs[:, :, COEFF_A[0], COEFF_A[1]]
    b = coeffs[:, :, COEFF_B[0], COEFF_B[1]]
    wanted = np.tile(pattern, (math.ceil(bh / TILE), math.ceil(bw / TILE)))[:bh, :bw]
    shortfall = STRENGTH * weight - (a - b) * wanted
    correction = np.where(active, np.maximum(0., shortfall), 0.) * wanted / 2
    modified = active & (np.abs(correction) > 1e-6)
    a += correction
    b -= correction

    updated = idctn(coeffs, axes=(-2, -1), norm='ortho').transpose(0, 2, 1, 3).reshape(hh, ww)
    pixels[:hh, :ww] += (updated - y)[:, :, None]
    result = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
    if not debug:
        return result

    debug_image = result.copy()
    for by, bx in np.argwhere(modified):
        x1, y1 = int(bx) * BLOCK, int(by) * BLOCK
        cv2.rectangle(debug_image, (x1, y1), (x1 + BLOCK - 1, y1 + BLOCK - 1), (255, 0, 0), 1)
    return result, debug_image


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def _basis(index: int) -> np.ndarray:
    x = np.arange(BLOCK, dtype=np.float32)
    return (math.sqrt(2 / BLOCK) * np.cos(math.pi * index * (x + .5) / BLOCK)).astype(np.float32)


_BASIS_2, _BASIS_3 = _basis(2), _basis(3)
_ANCHOR = (BLOCK // 2, BLOCK // 2)


def _differences(y: np.ndarray) -> np.ndarray:
    """DCT(2,3) minus DCT(3,2) of the 16x16 window centred at every pixel."""
    a = cv2.sepFilter2D(y, cv2.CV_32F, _BASIS_3, _BASIS_2, anchor=_ANCHOR)
    b = cv2.sepFilter2D(y, cv2.CV_32F, _BASIS_2, _BASIS_3, anchor=_ANCHOR)
    return a - b


def _weight_map(y: np.ndarray) -> np.ndarray:
    """Same 16x16 statistics as `_embedding_weight`, evaluated at EVERY candidate centre."""
    def box(v: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(v, cv2.CV_32F, (BLOCK, BLOCK), anchor=_ANCHOR, borderType=cv2.BORDER_REFLECT_101)

    mean = box(y)
    std = np.sqrt(np.maximum(0., box(y * y) - mean * mean))
    not_paper = np.clip((246. - mean) / 32., 0., 1.)
    textured = np.clip((std - 5.) / 20., 0., 1.)
    white = box((y >= 235).astype(np.float32))
    black = box((y <= 100).astype(np.float32))
    text_like = (white > .40) & (black > .015)
    weight = not_paper * textured * (~text_like)
    return np.where(weight >= DETECT_MIN_WEIGHT, weight, 0.).astype(np.float32)


def _fold(data: np.ndarray) -> np.ndarray:
    """Sum a (rows, cols) grid into one TILE x TILE tile, indexed by position modulo TILE."""
    h, w = data.shape
    padded = np.pad(data, ((0, -h % TILE), (0, -w % TILE)))
    return padded.reshape(-1, TILE, padded.shape[1] // TILE, TILE).sum(axis=(0, 2), dtype=np.float32)


def _fold_counts(h: int, w: int) -> np.ndarray:
    """How many grid positions of an (h, w) grid fall on each tile cell (independent of the data)."""
    per_row = np.bincount(np.arange(h) % TILE, minlength=TILE).astype(np.float32)
    per_col = np.bincount(np.arange(w) % TILE, minlength=TILE).astype(np.float32)
    return np.outer(per_row, per_col)


def _decode_tile(sums: np.ndarray, counts: np.ndarray, tile_y: int, tile_x: int, deep: bool = False) -> str | None:
    """Decode one folded tile at a candidate phase, retrying the least reliable bits."""
    sums = np.roll(sums, (-tile_y, -tile_x), axis=(0, 1)).ravel()
    counts = np.roll(counts, (-tile_y, -tile_x), axis=(0, 1)).ravel()
    soft = sums[DATA_INDICES] / np.maximum(1., counts[DATA_INDICES])
    bits = (soft > 0).astype(np.uint8) ^ MASK_BITS
    result = _decode_bits(bits)
    if result is not None:
        return result

    uncertain = np.argsort(np.abs(soft))[:16 if deep else 8]
    for i in uncertain:                                   # single-bit ambiguity
        alt = bits.copy()
        alt[i] ^= 1
        result = _decode_bits(alt)
        if result is not None:
            return result
    if deep:                                              # two-bit ambiguity
        for n, i in enumerate(uncertain[:12]):
            for j in uncertain[n + 1:12]:
                alt = bits.copy()
                alt[i] ^= 1
                alt[j] ^= 1
                result = _decode_bits(alt)
                if result is not None:
                    return result
    return None


def _detect(y: np.ndarray, deep: bool = False) -> tuple[str | None, float]:
    """Search DCT block alignment and periodic pilot phase.

    Returns (message or None, best pilot correlation score).
    """
    if min(y.shape) < MIN_DETECT_SIDE:
        return None, 0.
    diff = _differences(y)
    weight_map = _weight_map(y)
    h, w = diff.shape
    phases = 10 if deep else 2
    candidates = []
    for oy in range(BLOCK):
        rows = np.arange(BLOCK // 2 + oy, h - BLOCK // 2, BLOCK)
        if len(rows) < 2:
            continue
        for ox in range(BLOCK):
            cols = np.arange(BLOCK // 2 + ox, w - BLOCK // 2, BLOCK)
            if len(cols) < 2:
                continue
            weighted = diff[np.ix_(rows, cols)] * weight_map[np.ix_(rows, cols)]
            sums = _fold(weighted)
            counts = _fold_counts(len(rows), len(cols))
            correlation = np.fft.ifft2(np.fft.fft2(sums) * PILOT_FFT).real
            observations = np.fft.ifft2(np.fft.fft2(counts) * PILOT_ABS_FFT).real
            z = correlation / ((float(np.std(weighted)) + 1e-4) * np.sqrt(np.maximum(1, observations)))
            flat = z.ravel()
            for idx in np.argpartition(flat, -phases)[-phases:]:
                if deep or flat[idx] > 2.5:
                    sy, sx = np.unravel_index(idx, z.shape)
                    candidates.append((float(flat[idx]), sums, counts, sy, sx))
    candidates.sort(key=lambda c: c[0], reverse=True)
    for score, sums, counts, sy, sx in candidates[:120 if deep else 40]:
        result = _decode_tile(sums, counts, sy, sx, deep=deep)
        if result is not None:
            return result, score
    return None, candidates[0][0] if candidates else 0.


def _rotate(y: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return y
    h, w = y.shape
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2., (h - 1) / 2.), angle, 1.0)
    return cv2.warpAffine(y, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


# --------------------------------------------------------------------------- #
# Extraction search over scale / rotation hypotheses
# --------------------------------------------------------------------------- #
COMMON_SCALES = (1., 2., 1.333333333, 1.25, 1.428571429, 1.666666667,
                 1.176470588, 1.111111111, 1.5, .8, .666666667, .5, 1.2)
ANGLES = (0., -.5, .5, -1., 1.)
FINE_SCALES = sorted(np.arange(.75, 1.551, .005), key=lambda s: abs(math.log(s)))
MAX_SELECTED_SCALES = 8
MIN_SCALE_SEPARATION = .02


class _Searcher:
    """Runs hypotheses against one luminance image under a shared time budget."""

    def __init__(self, y: np.ndarray, seconds: float, started: float):
        self.y = y
        self.deadline = started + seconds
        self.last_progress = started
        self.ranked: list[tuple[float, float]] = []   # (pilot score, scale) from the fine scan

    @property
    def expired(self) -> bool:
        return time.monotonic() > self.deadline

    def check(self, upright: np.ndarray, scale: float, deep: bool = False, collect: bool = False) -> str | None:
        if self.expired:
            return None
        now = time.monotonic()
        if now - self.last_progress >= PROGRESS_INTERVAL:
            print('[extract] searching watermark...', file=sys.stderr)
            self.last_progress = now
        h, w = upright.shape
        width, height = round(w * scale), round(h * scale)
        if (min(height, width) < 128
                or (scale > 1 and max(height, width) > MAX_EXTRACT_SIDE)
                or height * width > MAX_PIXELS):
            return None
        if scale == 1:
            scaled = upright
        else:
            interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            scaled = cv2.resize(upright, (width, height), interpolation=interpolation)
        found, strength = _detect(scaled, deep=deep)
        if collect:
            self.ranked.append((strength, scale))
        return found

    def sweep(self, scales, *, angles=(0.,), deep=False, collect=False, offsets=(0.,)) -> str | None:
        """Try every (angle, scale, offset) combination in order; stop at the first hit."""
        for angle in angles:
            if self.expired:
                return None
            upright = _rotate(self.y, angle)
            for scale in scales:
                for offset in offsets:
                    if self.expired:
                        return None
                    result = self.check(upright, scale + offset, deep=deep, collect=collect)
                    if result is not None:
                        return result
        return None

    def best_scales(self) -> list[float]:
        selected: list[float] = []
        for _, scale in sorted(self.ranked, reverse=True):
            if all(abs(scale - old) >= MIN_SCALE_SEPARATION for old in selected):
                selected.append(float(scale))
            if len(selected) == MAX_SELECTED_SCALES:
                break
        return selected


def _stages(searcher: _Searcher) -> Iterator[str | None]:
    """Lazily run the search stages, cheapest first."""
    yield searcher.sweep(COMMON_SCALES, angles=ANGLES)                     # fast: common ratios, deskew
    yield searcher.sweep(COMMON_SCALES, angles=ANGLES, deep=True)          # deep: more candidates
    fine = [float(s) for s in FINE_SCALES if min(abs(s - c) for c in COMMON_SCALES) >= .003]
    yield searcher.sweep(fine, collect=True)                               # fine scale scan, no rotation
    best = searcher.best_scales()
    yield searcher.sweep(best, deep=True)                                  # promising scales, deep
    yield searcher.sweep(best, angles=ANGLES[1:], deep=True, offsets=(-.01, 0., .01))


def extract(rgb: np.ndarray) -> str | None:
    """Fast search, then a more exhaustive search without changing the watermark."""
    if rgb.shape[0] * rgb.shape[1] > MAX_PIXELS:
        raise ValueError(f'Image exceeds {MAX_PIXELS:,} pixels.')
    deadline_start = time.monotonic()
    y = _luma(rgb)
    if max(y.shape) > MAX_EXTRACT_SIDE:
        factor = MAX_EXTRACT_SIDE / max(y.shape)
        y = cv2.resize(y, (round(y.shape[1] * factor), round(y.shape[0] * factor)), interpolation=cv2.INTER_AREA)
    searcher = _Searcher(y, SEARCH_SECONDS, started=deadline_start)
    for result in _stages(searcher):
        if result is not None:
            return result
        if searcher.expired:
            break
    return None


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _run_embed(args: argparse.Namespace) -> int:
    fmt = _output_format(args.output)
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    _frame(args.message)                                  # validate the message early
    debug_path = args.output.with_name(args.output.stem + '_debug.png') if args.debug else None
    if debug_path is not None and debug_path.exists():
        raise FileExistsError(f'Debug output already exists: {debug_path}')

    result = embed(_load(args.input), args.message, debug=args.debug)
    image, debug_image = result if args.debug else (result, None)

    # Verify the exact bytes that will be written; never claim success otherwise.
    encoded = _encode_image(image, fmt)
    with Image.open(io.BytesIO(encoded)) as check_image:
        checked = np.asarray(check_image.convert('RGB'), dtype=np.uint8)
    found, _ = _detect(_luma(checked), deep=True)
    if found != args.message:
        raise ValueError('Embedding failed verification: output watermark cannot be recovered. No file saved.')

    _write_new(encoded, args.output)
    print(f'Watermark embedded: {args.output}')
    if debug_image is not None:
        _write_new(_encode_image(debug_image, 'PNG'), debug_path)
        print(f'Debug visualization: {debug_path}')
    return 0


def _run_extract(args: argparse.Namespace) -> int:
    result = extract(_load(args.input))
    if result is None:
        print('Watermark not found.', file=sys.stderr)
        return 1
    print(result)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Invisible image watermark (experimental print/scan resistance)')
    commands = parser.add_subparsers(dest='command', required=True)
    encoder = commands.add_parser('embed', help=f'Embed up to {MAX_MESSAGE_BYTES} UTF-8 bytes')
    encoder.add_argument('input', type=Path)
    encoder.add_argument('output', type=Path)
    encoder.add_argument('message')
    encoder.add_argument('--debug', action='store_true', help='Save a PNG with red borders around modified DCT blocks')
    encoder.set_defaults(run=_run_embed)
    decoder = commands.add_parser('extract', help='Read the hidden text; no original required')
    decoder.add_argument('input', type=Path)
    decoder.set_defaults(run=_run_extract)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        return args.run(args)
    except (ValueError, OSError, cv2.error, MemoryError, Image.DecompressionBombError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())