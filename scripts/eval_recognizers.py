"""Compares handwriting line recognizers on the same line crops (see scripts/prepare_gnhk.py).

Each recognizer reads every sampled crop; models run one at a time and are
unloaded in between, so they fit a 4GB GPU. Scores:
  cer / wer            exact, case-sensitive
  cer_norm             lowercase, punctuation removed, whitespace collapsed
  cer_corrected        exact CER after WordCorrector (no confidences, so it only
                       touches non-dictionary words)

Usage:
  $env:PYTHONPATH = "src"
  .venv\\Scripts\\python.exe scripts\\eval_recognizers.py --limit 300 --models paddle got trocr paddleocr_vl qwen3_vl
"""

import argparse
import datetime
import gc
import json
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import edit_distance, normalize  # noqa: E402

MAX_HEIGHT = 160
MAX_WIDTH = 1600
QWEN_PROMPT = "Transcribe the handwritten text in this image exactly as written. Output only the text, on one line."


def resize(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    scale = min(1.0, MAX_HEIGHT / h, MAX_WIDTH / w)
    return cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else crop


def to_pil(crop: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


class PaddleRec:
    name = "PP-OCRv6 medium rec (current fast path)"

    def __init__(self, models_dir):
        import ocr_pipeline  # noqa: F401  (points PaddleX's model cache at models/paddleocr)
        from paddleocr import TextRecognition

        self.model = TextRecognition(model_name="PP-OCRv6_medium_rec", device="cpu", enable_mkldnn=False)

    def read(self, crop):
        return self.model.predict(input=crop)[0]["rec_text"]


class GotOcr:
    name = "GOT-OCR2.0 (current fallback)"

    def __init__(self, models_dir):
        from ocr_pipeline.recognizer_specialist import GotOcrRecognizer

        self.model = GotOcrRecognizer(str(models_dir / "got_ocr2"), device="cuda", max_new_tokens=128, batch_size=1)

    def read(self, crop):
        return self.model.recognize(crop)


class TrOcr:
    name = "TrOCR-large-handwritten"

    def __init__(self, models_dir):
        from transformers import AutoImageProcessor, RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

        path = models_dir / "trocr_large_handwritten"
        # TrOCRProcessor.from_pretrained fails on this repo's legacy tokenizer
        # config under transformers 5; loading the tokenizer class directly works.
        tokenizer = RobertaTokenizer.from_pretrained(path)
        self.processor = TrOCRProcessor(image_processor=AutoImageProcessor.from_pretrained(path), tokenizer=tokenizer)
        self.model = VisionEncoderDecoderModel.from_pretrained(path, dtype=torch.float16).to("cuda").eval()

    @torch.inference_mode()
    def read(self, crop):
        pixel_values = self.processor(images=to_pil(crop), return_tensors="pt").pixel_values.to("cuda", torch.float16)
        ids = self.model.generate(pixel_values, max_new_tokens=96, use_cache=True)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class PaddleOcrVl:
    name = "PaddleOCR-VL-1.5"

    def __init__(self, models_dir):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        path = models_dir / "paddleocr_vl_1_5"
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()

    @torch.inference_mode()
    def read(self, crop):
        messages = [{"role": "user", "content": [{"type": "image", "image": to_pil(crop)}, {"type": "text", "text": "OCR:"}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return self.processor.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()


class Qwen3Vl:
    name = "Qwen3-VL-2B-Instruct (4-bit)"

    def __init__(self, models_dir):
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        path = models_dir / "qwen3_vl_2b"
        self.processor = AutoProcessor.from_pretrained(path)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, quantization_config=quant, device_map="cuda"
        ).eval()

    @torch.inference_mode()
    def read(self, crop):
        messages = [{"role": "user", "content": [{"type": "image", "image": to_pil(crop)}, {"type": "text", "text": QWEN_PROMPT}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        text = self.processor.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return " ".join(text.split())


RECOGNIZERS = {"paddle": PaddleRec, "got": GotOcr, "trocr": TrOcr, "paddleocr_vl": PaddleOcrVl, "qwen3_vl": Qwen3Vl}


def norm_loose(text: str) -> str:
    return normalize(re.sub(r"[^\w\s]", "", text.lower()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lines", default="benchmark/gnhk/lines")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--models", nargs="+", default=list(RECOGNIZERS), choices=list(RECOGNIZERS))
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="benchmark/results")
    args = parser.parse_args()

    lines_dir = Path(args.lines)
    labels = [json.loads(line) for line in (lines_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    random.Random(args.seed).shuffle(labels)
    labels = labels[: args.limit]
    crops = [resize(cv2.imread(str(lines_dir / label["file"]))) for label in labels]

    from ocr_pipeline.word_corrector import WordCorrector

    corrector = WordCorrector()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.datetime.now():%Y%m%d-%H%M%S}-recognizers.json"
    results = {}
    for key in args.models:
        load_start = time.perf_counter()
        recognizer = RECOGNIZERS[key](Path(args.models_dir))
        load_seconds = time.perf_counter() - load_start
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        recognizer.read(crops[0])  # warm-up

        outputs, start = [], time.perf_counter()
        for crop in crops:
            try:
                outputs.append(normalize(recognizer.read(crop)))
            except Exception as error:  # one bad crop shouldn't sink the whole comparison
                print(f"  {key}: {type(error).__name__}: {error}", flush=True)
                outputs.append("")
        seconds = (time.perf_counter() - start) / len(crops)
        corrected = [normalize(text) for text in corrector.correct_lines(outputs)]

        totals = dict(edits=0, chars=0, word_edits=0, words=0, loose_edits=0, loose_chars=0, corr_edits=0)
        for label, hyp, fixed in zip(labels, outputs, corrected):
            ref = normalize(label["text"])
            totals["edits"] += edit_distance(list(hyp), list(ref))
            totals["chars"] += len(ref)
            totals["word_edits"] += edit_distance(hyp.split(), ref.split())
            totals["words"] += len(ref.split())
            totals["loose_edits"] += edit_distance(list(norm_loose(hyp)), list(norm_loose(ref)))
            totals["loose_chars"] += len(norm_loose(ref))
            totals["corr_edits"] += edit_distance(list(fixed), list(ref))

        results[key] = {
            "name": recognizer.name,
            "cer": totals["edits"] / totals["chars"],
            "wer": totals["word_edits"] / totals["words"],
            "cer_norm": totals["loose_edits"] / max(totals["loose_chars"], 1),
            "cer_corrected": totals["corr_edits"] / totals["chars"],
            "seconds_per_line": seconds,
            "load_seconds": load_seconds,
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
            "samples": [
                {"file": label["file"], "ref": label["text"], "hyp": hyp, "corrected": fixed}
                for label, hyp, fixed in zip(labels, outputs, corrected)
            ],
        }
        r = results[key]
        print(f"{r['name']:<42} CER {r['cer']:.4f}  WER {r['wer']:.4f}  CER-norm {r['cer_norm']:.4f}  "
              f"CER+corrector {r['cer_corrected']:.4f}  {r['seconds_per_line']:.2f}s/line  "
              f"VRAM {r['peak_vram_mib']} MiB", flush=True)

        # Saved after every model so a later crash doesn't lose earlier results.
        out_path.write_text(json.dumps({"lines": len(labels), "seed": args.seed, "results": results}, indent=2), encoding="utf-8")
        del recognizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
