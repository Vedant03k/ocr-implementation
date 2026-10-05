"""Builds benchmark sets from the GNHK test split (real camera photos of English handwriting).

GNHK (Lee et al., ICDAR 2021) is CC BY 4.0: https://github.com/GoodNotes/GNHK-dataset
Annotations are word-level: {"text", "polygon": {x0..y3}, "line_idx", "type"}.
Words GNHK couldn't transcribe are tokens like "%math%" or "%SC%".

Writes:
  <out>/lines/    one crop per annotated line + labels.jsonl  (for scripts/eval_recognizers.py)
  <out>/pages/    full photos + .txt transcriptions            (for scripts/benchmark.py)
Lines containing a %...% token are skipped (no ground truth to score against);
on pages, those words are dropped from the reference.

Usage: python scripts/prepare_gnhk.py --src benchmark/downloads/gnhk/test --out benchmark/gnhk
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PAD = 12


def is_placeholder(word: str) -> bool:
    return word.startswith("%") and word.endswith("%")


def line_groups(words: list[dict]) -> list[list[dict]]:
    groups = defaultdict(list)
    for word in words:
        groups[word["line_idx"]].append(word)
    for group in groups.values():
        group.sort(key=lambda w: min(w["polygon"][f"x{i}"] for i in range(4)))
    return [groups[k] for k in sorted(groups)]


def crop_line(image: np.ndarray, words: list[dict]) -> np.ndarray:
    # A rectangular crop of a slanted handwritten line also catches pieces of the
    # lines above and below, which models then (correctly) read. Keep only the
    # line's own word outlines, slightly dilated, and fill the rest with paper colour.
    height, width = image.shape[:2]
    polys = [
        np.array([[w["polygon"][f"x{i}"], w["polygon"][f"y{i}"]] for i in range(4)], dtype=np.int32) for w in words
    ]
    points = np.vstack(polys)
    x1, y1 = np.maximum(points.min(axis=0) - PAD, 0)
    x2, y2 = np.minimum(points.max(axis=0) + PAD, [width, height])
    crop = image[y1:y2, x1:x2].copy()
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [p - [x1, y1] for p in polys], 255)
    mask = cv2.dilate(mask, np.ones((2 * PAD + 1, 2 * PAD + 1), np.uint8))
    outside = mask == 0
    if outside.any():
        crop[outside] = np.median(crop[outside], axis=0)
    return crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="benchmark/downloads/gnhk/test")
    parser.add_argument("--out", default="benchmark/gnhk")
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    lines_dir, pages_dir = out / "lines", out / "pages"
    lines_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    labels, skipped = [], 0
    for ann_path in sorted(src.glob("*.json")):
        image_path = ann_path.with_suffix(".jpg")
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        page_lines = []
        for idx, words in enumerate(line_groups(json.loads(ann_path.read_text(encoding="utf-8")))):
            texts = [w["text"] for w in words]
            page_lines.append(" ".join(t for t in texts if not is_placeholder(t)))
            if any(is_placeholder(t) for t in texts):
                skipped += 1
                continue
            crop_name = f"{ann_path.stem}_L{idx:02d}.png"
            cv2.imwrite(str(lines_dir / crop_name), crop_line(image, words))
            labels.append({"file": crop_name, "text": " ".join(texts), "page": ann_path.stem, "line": idx})

        shutil.copy(image_path, pages_dir / image_path.name)
        (pages_dir / f"{ann_path.stem}.txt").write_text(
            "\n".join(line for line in page_lines if line) + "\n", encoding="utf-8"
        )

    with open(lines_dir / "labels.jsonl", "w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")
    print(f"{len(labels)} line crops ({skipped} skipped for %...% tokens), "
          f"{len(list(pages_dir.glob('*.jpg')))} pages -> {out}")


if __name__ == "__main__":
    main()
