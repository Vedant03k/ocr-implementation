"""Latency + accuracy benchmark for the local OCR pipeline (src/ocr_pipeline/api.py).

Data folder: images (.png/.jpg/.jpeg/.webp), each with a same-named .txt
holding the correct transcription, one text line per line in reading order.

Runs the same per-image code path as the /api/ocr endpoint, timing each model
stage, and scores the text the GUI would show at the given threshold:
  paddleocr       PaddleOCR text for every line
  routed          GOT-OCR2.0 text for lines below the threshold, PaddleOCR otherwise
  routed_cleaned  the LLM-cleaned version of `routed` (what the GUI shows by default)
CER/WER are edit distance over the whole page (lines joined by spaces),
case-sensitive, summed over all images.

Usage (from the repo root, models must be downloaded):
  $env:PYTHONPATH = "src"
  .venv\\Scripts\\python.exe scripts\\benchmark.py --data benchmark\\synthetic --label baseline
"""

import argparse
import datetime
import inspect
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
# Only the first method found per stage is timed, so a method that delegates to
# another (recognize -> recognize_batch) isn't counted twice.
STAGES = {
    "detector": ["detect"],
    "specialist": ["recognize_batch", "recognize"],
    "cleaner": ["clean_lines"],
    "corrector": ["correct_lines"],
}


def edit_distance(a: list, b: list) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def normalize(text: str) -> str:
    return " ".join(text.split())


class StageTimer:
    def __init__(self, sync):
        self.sync = sync
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    def wrap(self, obj, method_name: str, stage: str):
        original = getattr(obj, method_name)

        def timed(*args, **kwargs):
            self.sync()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            self.sync()
            self.seconds[stage] += time.perf_counter() - start
            self.calls[stage] += 1
            return result

        setattr(obj, method_name, timed)

    def reset(self):
        self.seconds.clear()
        self.calls.clear()


def page_texts(detections, image_height: int, threshold: float) -> dict[str, str]:
    ordered = sorted(detections, key=lambda d: (round(d.y * image_height / 15), d.x))
    variants = defaultdict(list)
    for d in ordered:
        escalate = d.rawConfidence < threshold and d.specialistText is not None
        routed = d.specialistText if escalate else d.candidateText
        cleaned = d.cleanedSpecialistText if escalate else d.cleanedCandidateText
        variants["paddleocr"].append(d.candidateText)
        variants["routed"].append(routed)
        variants["routed_cleaned"].append(cleaned if cleaned is not None else routed)
    return {name: normalize(" ".join(parts)) for name, parts in variants.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="folder of images + same-named .txt transcriptions")
    parser.add_argument("--label", default="run", help="name for this run, used in the output filename")
    parser.add_argument("--threshold", type=float, default=None, help="routing threshold (default: config value)")
    parser.add_argument("--out-dir", default="benchmark/results")
    args = parser.parse_args()

    images = sorted(p for p in Path(args.data).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    images = [p for p in images if p.with_suffix(".txt").exists()]
    if not images:
        sys.exit(f"no image + .txt pairs found in {args.data}")

    import cv2
    import torch

    load_start = time.perf_counter()
    from ocr_pipeline import api

    load_seconds = time.perf_counter() - load_start
    threshold = args.threshold if args.threshold is not None else api.config["router"]["confidence_threshold"]

    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    timer = StageTimer(sync)
    for attr, methods in STAGES.items():
        obj = getattr(api, attr, None)
        method = next((m for m in methods if obj is not None and hasattr(obj, m)), None)
        if method:
            timer.wrap(obj, method, attr)

    accepts_threshold = "threshold" in inspect.signature(api._detections_for_frame).parameters
    frame_kwargs = {"threshold": threshold} if accepts_threshold else {}
    api._detections_for_frame(cv2.imread(str(images[0])), "warmup", **frame_kwargs)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    per_image = []
    totals = defaultdict(lambda: {"edits_char": 0, "ref_char": 0, "edits_word": 0, "ref_word": 0})
    for path in images:
        image = cv2.imread(str(path))
        reference = normalize(path.with_suffix(".txt").read_text(encoding="utf-8"))
        timer.reset()
        start = time.perf_counter()
        detections = api._detections_for_frame(image, "d", **frame_kwargs)
        sync()
        seconds = time.perf_counter() - start

        texts = page_texts(detections, image.shape[0], threshold)
        scores = {}
        for name, hypothesis in texts.items():
            char_edits = edit_distance(list(hypothesis), list(reference))
            word_edits = edit_distance(hypothesis.split(), reference.split())
            scores[name] = {
                "cer": char_edits / max(len(reference), 1),
                "wer": word_edits / max(len(reference.split()), 1),
            }
            totals[name]["edits_char"] += char_edits
            totals[name]["ref_char"] += len(reference)
            totals[name]["edits_word"] += word_edits
            totals[name]["ref_word"] += len(reference.split())

        per_image.append({
            "image": path.name,
            "seconds": seconds,
            "stage_seconds": dict(timer.seconds),
            "stage_calls": dict(timer.calls),
            "lines_detected": len(detections),
            "lines_escalated": sum(d.rawConfidence < threshold for d in detections),
            "reference": reference,
            "texts": texts,
            "scores": scores,
        })
        print(f"{path.name}: {seconds:6.2f}s  lines={len(detections)}  "
              + "  ".join(f"{n} CER={s['cer']:.3f}" for n, s in scores.items()))

    accuracy = {
        name: {"cer": t["edits_char"] / max(t["ref_char"], 1), "wer": t["edits_word"] / max(t["ref_word"], 1)}
        for name, t in totals.items()
    }
    stage_mean = defaultdict(float)
    for result in per_image:
        for stage, value in result["stage_seconds"].items():
            stage_mean[stage] += value / len(per_image)
    latencies = sorted(r["seconds"] for r in per_image)

    gpu = None
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        peak_reserved = torch.cuda.max_memory_reserved()
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_vram_mib": round(total_vram / 2**20),
            "peak_reserved_mib": round(peak_reserved / 2**20),
            # On Windows, CUDA allocations past dedicated VRAM silently spill into
            # shared system memory (sysmem fallback) instead of failing.
            "exceeds_dedicated_vram": peak_reserved > total_vram,
        }

    summary = {
        "label": args.label,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "data": args.data,
        "images": len(per_image),
        "threshold": threshold,
        "model_load_seconds": load_seconds,
        "latency_seconds": {
            "mean": sum(latencies) / len(latencies),
            "median": latencies[len(latencies) // 2],
            "max": latencies[-1],
        },
        "mean_stage_seconds": dict(stage_mean),
        "accuracy": accuracy,
        "gpu": gpu,
        "config": api.config,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{args.label}.json"
    out_path.write_text(json.dumps({"summary": summary, "per_image": per_image}, indent=2), encoding="utf-8")

    print(f"\n== {args.label}: {len(per_image)} images, threshold {threshold} ==")
    lat = summary["latency_seconds"]
    print(f"latency  mean {lat['mean']:.2f}s  median {lat['median']:.2f}s  max {lat['max']:.2f}s")
    print("stages   " + "  ".join(f"{k} {v:.2f}s" for k, v in stage_mean.items()))
    for name, a in accuracy.items():
        print(f"{name:<15} CER {a['cer']:.4f}  WER {a['wer']:.4f}")
    if gpu:
        print(f"gpu      peak {gpu['peak_reserved_mib']} MiB of {gpu['total_vram_mib']} MiB"
              + ("  ** exceeds dedicated VRAM (spilling to system RAM) **" if gpu["exceeds_dedicated_vram"] else ""))
    print(f"saved    {out_path}")


if __name__ == "__main__":
    main()
