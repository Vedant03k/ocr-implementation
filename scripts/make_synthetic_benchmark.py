"""Generates a small synthetic handwriting-style benchmark set for scripts/benchmark.py.

Renders multi-line notes in Windows' handwriting-style fonts with slight
rotation, blur and noise. This only checks that the harness works and gives a
rough baseline: font-rendered "handwriting" is far cleaner than real cursive,
so decisions must be validated on real images (put those in benchmark/real/).

Usage: python scripts/make_synthetic_benchmark.py [--out benchmark/synthetic] [--count 10]
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = ["Inkfree.ttf", "segoesc.ttf", "segoepr.ttf", "LHANDW.TTF", "BRADHITC.TTF", "arial.ttf"]

LINES = [
    "Theme: to build seamless infrastructure for",
    "our industry to find out what works",
    "Project name: Shadow Sentinel",
    "please rush this one, job site closes Friday",
    "call me if the delivery is delayed",
    "received by J.T. - ok to ship",
    "restock by Friday, low on M8 bolts",
    "check pallet weight before loading",
    "hire two more forklift operators",
    "review with Marcus on Tuesday at 9am",
    "Subtotal: $842.00 including tax",
    "meeting notes for the quarterly review",
    "the quick brown fox jumps over the lazy dog",
    "send the invoice to accounts before noon",
    "remember to update the safety checklist",
    "budget approved for the new warehouse",
    "follow up with the supplier about pricing",
    "deadline moved to the end of next week",
]


def render(lines: list[str], font_path: str, rng: random.Random) -> Image.Image:
    size = rng.randint(34, 44)
    font = ImageFont.truetype(font_path, size)
    line_height = int(size * 1.9)
    width = 1200
    height = line_height * len(lines) + 120
    background = tuple(rng.randint(235, 250) for _ in range(3))
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    ink = tuple(rng.randint(10, 60) for _ in range(3))
    for i, line in enumerate(lines):
        draw.text((60 + rng.randint(-10, 10), 60 + i * line_height), line, fill=ink, font=font)

    image = image.rotate(rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=True, fillcolor=background)
    image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))
    pixels = np.asarray(image).astype(np.int16)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 6, pixels.shape)
    return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmark/synthetic")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        font = FONTS[i % len(FONTS)]
        lines = rng.sample(LINES, rng.randint(3, 6))
        render(lines, f"C:/Windows/Fonts/{font}", rng).save(out / f"synthetic_{i:02d}.png")
        (out / f"synthetic_{i:02d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.count} images to {out}")


if __name__ == "__main__":
    main()
