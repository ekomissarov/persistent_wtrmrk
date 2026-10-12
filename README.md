# Invisible Watermark — Python

> **AI-generated project:** This project's code was written entirely by AI. It is an experimental implementation and has not undergone an independent security or robustness audit.

A minimal Python command-line tool that embeds a short, invisible watermark into an image and extracts it without needing the original image (**blind watermarking**).

The project uses **one fixed algorithm**. There are no secret keys, encryption options, or configuration modes.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Installation

```bash
python -m pip install -r requirements.txt
```

## Usage

Embed a message:

```bash
python watermark.py embed photo.jpg marked.png "Photo archive"
```

Extract the message:

```bash
python watermark.py extract marked.png
```

Expected output:

```text
Photo archive
```

The original image is not required for extraction.

### Input limitations

- **Image formats:** PNG and JPEG.
- **Message length:** 1–16 bytes encoded as UTF-8 (not necessarily 16 characters).
- **Minimum image size:** 320 × 320 pixels.
- **Recommended output:** PNG, to avoid additional JPEG compression.
- **Output files:** Existing files are not overwritten.

## How it works

The algorithm combines frequency-domain embedding, repeated data, synchronization markers, and error correction:

1. The image is processed in **16 × 16 pixel blocks** using the Discrete Cosine Transform (DCT).
2. Each bit is embedded by modifying the relationship between two mid-frequency DCT coefficients.
3. The watermark pattern is repeated in **320 × 320 pixel tiles**, improving the chance of recovery after partial cropping.
4. Each tile contains **160 synchronization symbols** to help locate the watermark after transformations.
5. The payload contains up to **16 bytes of text**, a length byte, and a **CRC32 checksum**. The resulting 168 bits are encoded into **240 bits using Hamming(15,11)**.
6. Extraction automatically searches for changes in scale and small rotations.

CRC32 detects accidental corruption. It is **not** a digital signature or an authorship proof.

## Robustness

The prototype has been tested with digital transformations, including:

- JPEG compression
- Resizing
- Partial cropping
- Small rotations
- Combinations of resizing and rotation
- **Simulated** print-scan effects (resolution loss, blur, tone changes, and noise)

Recovery is **not guaranteed** for every image or transformation.

### Print-scan limitation

**Physical print-scan robustness has not been validated.** Simulating printing and scanning digitally does not reproduce all effects of real printers, paper, and scanners, including halftoning, paper texture, nonuniform deformation, and perspective distortion.

## Tests

Run the automated test suite:

```bash
python -m unittest -v test_watermark.py
```

## Security and limitations

- **No cryptographic authentication:** The watermark does not prove who created the image.
- **No secret key:** The embedding scheme is public. A knowledgeable attacker may be able to overwrite or remove the mark.
- **No guarantee against intentional removal:** Resistance to ordinary image processing is different from resistance to targeted attacks.
- **Possible visual artifacts:** Embedding changes pixel values and may be noticeable in smooth image regions.
- **Limited geometric robustness:** Severe cropping, rotation, perspective distortion, or repeated processing may prevent extraction.
- **Extraction cost:** Searching for geometric alignment can take tens of seconds on difficult images.

## Project status

**Experimental prototype.** The goal is to study how reliably a short invisible message can survive image transformations, especially a future physical **print → scan** test, while maintaining acceptable visual quality. The implementation is not intended for legal proof of ownership or high-assurance content authentication.
