"""
Steganography Dataset Encoder
==============================
Encodes hidden messages into all images in a directory using LSB steganography.
Produces a labeled dataset (clean=0, stego=1) ready for ML/QML training.

Usage:
    python stego_encoder.py --input ./images --output ./dataset
    python stego_encoder.py --input ./images --output ./dataset --payloads 25 50 75
"""

import os
import argparse
import random
import string
import numpy as np
import pandas as pd
import shutil
from PIL import Image
from pathlib import Path


# ─────────────────────────────────────────────
#  Core LSB Encoder / Decoder
# ─────────────────────────────────────────────

def text_to_bits(text: str) -> str:
    """Convert a string to a binary bit string."""
    bits = ''.join(format(ord(c), '08b') for c in text)
    return bits


def bits_to_text(bits: str) -> str:
    """Convert a binary bit string back to a string."""
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(c, 2)) for c in chars if len(c) == 8)


def embed_lsb(image: np.ndarray, message: str) -> np.ndarray:
    """
    Embed a message into an image using LSB substitution.
    Prepends a 32-bit length header so the decoder knows when to stop.
    """
    img = image.copy().astype(np.uint8)
    flat = img.flatten()

    # Build bit payload: 32-bit length header + message bits
    msg_bits = text_to_bits(message)
    length_bits = format(len(msg_bits), '032b')
    full_bits = length_bits + msg_bits

    if len(full_bits) > len(flat):
        raise ValueError(
            f"Message too large: needs {len(full_bits)} bits, "
            f"image only has {len(flat)} pixels."
        )

    for i, bit in enumerate(full_bits):
        flat[i] = (flat[i] & 0xFE) | int(bit)   # clear LSB, set new bit

    return flat.reshape(img.shape)


def extract_lsb(image: np.ndarray) -> str:
    """Extract a hidden message from an LSB-encoded image."""
    flat = image.flatten()
    lsb_stream = ''.join(str(p & 1) for p in flat)

    # Read 32-bit length header first
    length = int(lsb_stream[:32], 2)
    msg_bits = lsb_stream[32: 32 + length]
    return bits_to_text(msg_bits)


# ─────────────────────────────────────────────
#  Payload Generator
# ─────────────────────────────────────────────

def generate_message(image: np.ndarray, payload_pct: float) -> str:
    """
    Generate a random ASCII message that fills exactly `payload_pct`%
    of the image's available LSB capacity (excluding the 32-bit header).
    """
    total_pixels = image.size                   # H × W × C
    header_bits  = 32
    usable_bits  = total_pixels - header_bits
    target_bits  = int(usable_bits * (payload_pct / 100.0))
    num_chars    = max(1, target_bits // 8)

    chars = string.ascii_letters + string.digits + string.punctuation + ' '
    return ''.join(random.choices(chars, k=num_chars))


# ─────────────────────────────────────────────
#  Dataset Builder
# ─────────────────────────────────────────────

SUPPORTED_EXTS = {'.png', '.bmp', '.tiff', '.tif', '.jpg', '.jpeg'}

def build_dataset(
    input_dir: str,
    output_dir: str,
    payloads: list[float] = [25.0, 50.0, 75.0],
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)

    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    clean_dir = output_path / 'clean'
    stego_dir = output_path / 'stego'
    clean_dir.mkdir(parents=True, exist_ok=True)
    stego_dir.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTS
    ]

    if not image_files:
        print(f"[ERROR] No supported images found in '{input_dir}'.")
        print(f"        Supported formats: {', '.join(SUPPORTED_EXTS)}")
        return

    print(f"\n{'='*55}")
    print(f"  Steganography Dataset Encoder")
    print(f"{'='*55}")
    print(f"  Source images  : {len(image_files)}")
    print(f"  Payload levels : {payloads}")
    print(f"  Output dir     : {output_path}")
    print(f"{'='*55}\n")

    records = []
    errors  = []

    for img_path in sorted(image_files):
        # ── Load image ──────────────────────────────────────
        try:
            pil_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"  [SKIP] {img_path.name} — could not open: {e}")
            errors.append({'file': img_path.name, 'reason': str(e)})
            continue

        img_array = np.array(pil_img)
        stem      = img_path.stem

        # ── Save clean copy ─────────────────────────────────
        clean_out = clean_dir / f"{stem}_clean.png"
        pil_img.save(str(clean_out), format='PNG')
        records.append({
            'filename' : clean_out.name,
            'path'     : str(clean_out),
            'label'    : 0,
            'label_str': 'clean',
            'payload'  : 0.0,
            'source'   : img_path.name
        })

        # ── Embed at each payload level ──────────────────────
        for pct in payloads:
            try:
                message     = generate_message(img_array, pct)
                stego_array = embed_lsb(img_array, message)

                out_name = f"{stem}_stego_p{int(pct)}.png"
                out_path = stego_dir / out_name
                Image.fromarray(stego_array.astype(np.uint8)).save(
                    str(out_path), format='PNG'
                )

                # Verify round-trip
                recovered = extract_lsb(np.array(Image.open(str(out_path))))
                ok = recovered == message

                records.append({
                    'filename'  : out_name,
                    'path'      : str(out_path),
                    'label'     : 1,
                    'label_str' : 'stego',
                    'payload'   : pct,
                    'source'    : img_path.name
                })

                status = 'OK' if ok else 'VERIFY_FAIL'
                print(f"  [{status}] {img_path.name}  →  {out_name}  "
                      f"({pct}% payload, {len(message)} chars)")

            except Exception as e:
                print(f"  [ERR]  {img_path.name} @ {pct}% — {e}")
                errors.append({'file': img_path.name, 'payload': pct, 'reason': str(e)})

    # ── Save labels CSV ──────────────────────────────────────
    df = pd.DataFrame(records)
    csv_path = output_path / 'labels.csv'
    df.to_csv(csv_path, index=False)

    # ── Summary ──────────────────────────────────────────────
    clean_count = len(df[df['label'] == 0])
    stego_count = len(df[df['label'] == 1])

    print(f"\n{'='*55}")
    print(f"  Done!")
    print(f"  Clean images   : {clean_count}")
    print(f"  Stego images   : {stego_count}")
    print(f"  Total samples  : {len(df)}")
    print(f"  Labels CSV     : {csv_path}")
    if errors:
        print(f"  Errors/skipped : {len(errors)}")
    print(f"{'='*55}\n")

    if errors:
        err_df = pd.DataFrame(errors)
        err_path = output_path / 'errors.csv'
        err_df.to_csv(err_path, index=False)
        print(f"  Error log saved to: {err_path}\n")

    return df


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='LSB Steganography Dataset Encoder'
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Directory containing clean source images'
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help='Directory to write the labeled dataset into'
    )
    parser.add_argument(
        '--payloads', '-p', nargs='+', type=float,
        default=[25.0, 50.0, 75.0],
        help='Payload percentages to embed (default: 25 50 75)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build_dataset(
        input_dir  = args.input,
        output_dir = args.output,
        payloads   = args.payloads,
        seed       = args.seed
    )