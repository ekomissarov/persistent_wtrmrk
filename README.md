# Document-aware image watermark — v4 (research prototype)

> **AI-generated code:** This project's code was written entirely by AI. It has not received an independent security, quality, or robustness audit.

A simple public, blind watermark CLI for **synthetic document images**. One fixed algorithm, no key, no signatures, no configuration flags. It embeds up to **16 UTF-8 bytes** and attempts to recover them without the original image.

```bash
python -m pip install -r requirements.txt
python watermark.py embed synthetic_document.png marked.png "DEMO-2026"
python watermark.py extract marked.png
python -m unittest -v test_watermark.py
```

## Algorithm

Reuses a periodic 16×16-block DCT pattern on a 20×20-block tile with public synchronization markers, Hamming(15,11) coding, and CRC32. Unlike v3, **embedding strength depends on local texture and brightness**. The algorithm tries to avoid blank paper and obvious dark-on-white text, while favoring photographic or textured regions. Synchronization and recovery remain the same as in v3.

This heuristic is **not** a text detector and can misclassify areas. It does not identify or guarantee protection for MRZs, QR codes, signatures, security patterns, or sensitive fields. It must **not** be applied to actual identification documents without independent verification of document integrity.

## What has actually been tested

Synthetic composite sample: recovery after JPEG quality 70, 25% crop on each side, 75% resize, and a **digital approximation** of print/scan (blur, noise, brightness shift, resampling). This is not evidence of physical print/scan reliability.

On a completely white document with just black text, this version may embed **nothing at all**. This is a deliberate choice to favor readability over false claims of reliability. It is a meaningful limitation: this prototype is useful for mixed photographic/textured pages, not all document types.

## Limitations

- Not validated on any physical printer and scanner.
- No cryptographic authentication or proof of authorship; a public watermark can be overwritten.
- No guarantees against deliberate removal, perspective distortion, severe cropping, binary black/white thresholding, or OCR-based reconstruction.
- Flat regions are skipped: sufficient textured area must remain for decoding.
- Fixed 320×320 spatial repetition and 16-byte payload constrain capacity and robustness.
- Python extraction may be slow on difficult/unmarked images.
- The synthetic sample is **not an official document**.

Please test using fake IDs/certificates or other synthetic mockups, never alter real passports or IDs for verification workflows.
