# Document Watermark — Python

> **AI-generated project:** All application code in this project was written by AI. This is an experimental implementation; it has not undergone an independent security, document-integrity, or robustness audit.

A minimal command-line tool for embedding a short, visually unobtrusive watermark in raster images and recovering it **without the original image** (*blind watermarking*). The main research target is scans of complex documents (images with photos, textured backgrounds, stamps, and text), particularly recovery after **printing and scanning**.

**Status:** Experimental. Physical print → scan robustness has **not** been validated. Successful digital tests do not guarantee recovery on a real scanned document. Do not modify genuine identity documents or their security features for official use.

## Project specification

| Item | Specification |
| --- | --- |
| Interface | Python CLI: `embed`, `extract`; optional `--debug` for embedding |
| Algorithm | One fixed, public, document-aware blind DCT watermark |
| Inputs | PNG and JPEG, read through Pillow |
| Outputs | PNG or JPEG; existing files are not overwritten |
| Payload | 1–16 **UTF-8 bytes**, not necessarily 16 characters |
| Minimum image size for embedding | 320 × 320 pixels |
| Input image limit | 50,000,000 pixels |
| DCT blocks | 16 × 16 pixels |
| Repeated spatial tile | 20 × 20 blocks = 320 × 320 pixels |
| DCT coefficient pair | `(2, 3)` and `(3, 2)` |
| Maximum embedding margin | `STRENGTH = 115.0`, modulated by local content |
| Frame | 1-byte length + 16-byte padded message + 6-byte BLAKE2s checksum |
| Error correction | Hamming(15,11), yielding 255 coded bits |
| Synchronization | 145 public pilot positions per 400-position tile |
| Pattern generation | Fixed `SEED = 0x57A73`, SHA-256-derived deterministic layout |
| Extraction | Searches block alignments, tile phases, common/fine scales and small rotations |
| Extraction budget | `SEARCH_SECONDS = 110` seconds, checked between search hypotheses (not a strict hard timeout) |
| Security | No encryption, secret key, digital signature, or proof of authorship |

The watermark format in this version is not compatible with earlier prototypes; backward compatibility is not a project requirement.

## Requirements and installation

- Python 3.10 or newer
- Dependencies in `requirements.txt` (NumPy, SciPy, OpenCV, Pillow)

```bash
python -m pip install -r requirements.txt
```

## Usage

Embed text in an image:

```bash
python watermark.py embed document.png marked.png "DOC-2026"
```

Extract it without the original:

```bash
python watermark.py extract marked.png
```

Expected output:

```text
DOC-2026
```

Visualize where embedding modified DCT coefficients:

```bash
python watermark.py embed document.png marked.png "DOC-2026" --debug
```

This creates `marked.png` and `marked_debug.png`. The debug image draws **1-pixel red borders** around blocks with nonzero DCT coefficient corrections. These borders are for inspection only and are **not** part of the real watermark. They do not indicate guaranteed recovery from those regions.

If no suitable blocks exist or the generated output fails an immediate decoding check, `embed` returns an error instead of claiming success. The check runs on the selected output encoding (PNG or JPEG), but does **not** prove resilience to later transformations.

## Algorithm description

### 1. Message framing and redundancy

The input message is encoded in UTF-8 (up to 16 bytes), padded to 16 bytes and prefixed with its length. A **48-bit BLAKE2s checksum** is computed over this header to detect accidental misdecoding; it is **not keyed authentication**.

```text
Length                1 byte
Padded UTF-8 text    16 bytes
BLAKE2s checksum      6 bytes
----------------------------
Frame                23 bytes = 184 bits
Hamming(15,11)       17 codewords × 15 = 255 bits
Tile positions       255 data + 145 pilots = 400
```

Hamming coding can correct one bit error in each 15-bit codeword under its usual assumptions. It cannot necessarily recover arbitrarily missing or corrupted blocks.

### 2. Public repeating pattern

The code deterministically derives data positions, pilot positions, pilot signs and payload masking bits from the fixed public seed using SHA-256. A 20 × 20-block pattern repeats spatially throughout the image. Its pilots support synchronization after cropping and other transformations.

The seed is **not secret**; the pattern is reproducible by anyone with the code.

### 3. Content-adaptive embedding

The image is processed as luminance in 16 × 16 blocks. A local weight uses brightness, texture and a simple text-like-block heuristic. The implementation largely avoids white paper, high-contrast text-like regions and low-texture regions; this is **not OCR**, and it does not reliably identify protected document elements.

For eligible blocks, the algorithm adjusts the difference between DCT coefficients `(2, 3)` and `(3, 2)` to represent a bit. The weight modulates the embedding margin, trading signal strength against visual artifacts. Some documents may contain too few suitable blocks to embed a recoverable mark.

### 4. Blind extraction

The decoder estimates coefficient differences and a content-dependent weight map without consulting the original. It folds observations across repeated tiles, correlates with the public pilots to estimate phase and alignment, and reconstructs bits with Hamming error correction and checksum validation.

Extraction tries common scales and minor deskew angles first, then a deeper search with more synchronization candidates and ambiguous-bit hypotheses. If the input is large, the decoder may downscale it to a maximum side length of 3,600 pixels; downscaling itself may weaken the watermark. A progress message is printed during longer searches.

## Goals and acceptance criteria

The project is focused on **document images**, not on maximizing robustness for photographs alone. The intended priorities are:

1. **Reliable clean-image round-trip:** a successful `embed` must produce a file from which the embedded message can immediately be recovered.
2. **Low visibility:** minimize regular patterns and preserve legibility and appearance; evaluate artifacts around text, photographs, stamps and backgrounds.
3. **Digital robustness:** evaluate JPEG recompression, resizing, cropping, small rotations and their combinations.
4. **Physical print → scan robustness:** test on printed synthetic document pages scanned at different resolutions, with real alignment and printer/scanner artifacts.
5. **Operational reliability:** handle reasonably large scans, report failures clearly and keep extraction time observable.

These are **research goals**, not performance guarantees. In particular, reliable recovery after losing half of a document has not been established generally.

## Testing methodology

Use synthetic document layouts with text, a photograph, patterns and stamps rather than real passports or identity cards. Test the following separately:

| Test | What to measure |
| --- | --- |
| Clean PNG / JPEG round-trip | Exact message recovery after embedding |
| JPEG recompression | Recovery rate across quality levels |
| Resizing | Recovery rate across several downscale and upscale ratios |
| Cropping | Recovery rate for different crop areas and locations |
| Rotation / combined attacks | Recovery after small deskew and resize combinations |
| Physical print → scan | Recovery across printers, paper, resolutions and scanner settings |
| Visual fidelity | Side-by-side inspection, PSNR/SSIM where meaningful, legibility |
| Negative controls | False detections on images that were never watermarked |
| Runtime | Time and memory consumed, including unsuccessful searches |

A single successful test does not demonstrate reliability. Report results over multiple document layouts and image transformations. Digital print-scan simulations are useful for development but **cannot replace physical tests**.

## Limitations and safety

- **No authorship proof:** the watermark identifies only recoverable embedded text; it does not verify who created or owns the image.
- **No resistance guarantee against removal:** the publicly known algorithm may be intentionally overwritten or disrupted.
- **Sparse content:** pages consisting mostly of white paper and black text may have insufficient eligible blocks; embedding may fail.
- **Geometric sensitivity:** heavy cropping, large rotations, perspective warps, binarization and repeated scanning may defeat extraction.
- **Visible artifacts:** especially around complex high-contrast edges or in regions where an embedded signal becomes perceptible.
- **Document integrity:** the algorithm does not explicitly protect machine-readable zones, barcodes, QR codes, security patterns or text. It should not be used to alter genuine identity documents or relied on for their authenticity checks.
- **Runtime:** the 110-second budget is approximate because the deadline is checked between search operations; a single expensive operation can overrun it.
- **File handling:** input images are decoded in memory; the 50 MP limit does not imply a strict RAM limit. Output uses exclusive file creation and is not overwritten.

## Project status

**Research prototype — one fixed algorithm, no plugin architecture, no cryptographic features and no backward-compatibility layer.** The next meaningful validation step is an end-to-end test on **printed and rescanned synthetic documents**, including crop and resize cases that previously failed.
