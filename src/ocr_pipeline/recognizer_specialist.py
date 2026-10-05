import re

import cv2
import numpy as np
import torch
from PIL import Image


_REPEATED_RUN = re.compile(r"(?<!\S)(\S+)(?:\s+\1(?!\S)){2,}")


def collapse_repeats(text: str) -> str:
    """Cuts a decoding loop ("Ave Ave Ave Ave ...") down to one occurrence.

    A word written three or more times in a row on one handwritten line is far
    rarer than the model getting stuck, and one loop can wreck a whole page.
    """
    return _REPEATED_RUN.sub(r"\1", text)


# On a blank crop the model answers in prose instead of returning nothing.
_REFUSAL = re.compile(r"\b(image|picture)\b.*\b(blurry|unclear|no (visible |readable )?text|cannot|unable)\b", re.I)
# Maths operators and LaTeX: on handwriting these are misreads of letters
# ("∆V-11-38" for "LV-11-38"), carried over from formula training data.
# Arrows are kept; people do draw them in notes.
_MATH = re.compile(r"[∀-⏿]|\\[()\[\]]|\\[a-z]+\b")


def usable(text: str) -> str:
    """Returns the text, or "" when it is a refusal or a formula-style misread,
    so the caller keeps PaddleOCR's reading for that line instead."""
    return "" if _REFUSAL.search(text) or _MATH.search(text) else text


def _resolve_device(device: str | None) -> str:
    if device is None or (device.startswith("cuda") and not torch.cuda.is_available()):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class GotOcrRecognizer:
    name = "got_ocr2"

    def __init__(
        self,
        model_dir: str,
        device: str | None = None,
        max_new_tokens: int = 128,
        batch_size: int = 4,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = _resolve_device(device)
        # A single text line is well under 128 tokens; a larger cap only costs time
        # when GOT gets stuck repeating itself on a blank or tiny crop.
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.model = AutoModelForImageTextToText.from_pretrained(model_dir, device_map=self.device)
        self.processor = AutoProcessor.from_pretrained(model_dir)

    def recognize(self, crop: np.ndarray) -> str:
        return self.recognize_batch([crop])[0]

    def recognize_batch(self, crops: list[np.ndarray]) -> list[str]:
        texts: list[str] = []
        for start in range(0, len(crops), self.batch_size):
            images = [Image.fromarray(crop[:, :, ::-1]) for crop in crops[start : start + self.batch_size]]
            inputs = self.processor(images, return_tensors="pt").to(self.device)
            generate_ids = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=self.max_new_tokens,
            )
            decoded = self.processor.batch_decode(
                generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            texts.extend(text.strip() for text in decoded)
        return texts


class PaddleOcrVlRecognizer:
    """PaddleOCR-VL-1.5 (0.9B vision-language model) reading one text line per crop.

    On 300 real handwritten lines (GNHK) it had half GOT-OCR2.0's character
    error rate (10.2% vs 21.0%), at 1.1s vs 1.9s per line and 1.8GB vs 3.4GB VRAM.
    """

    name = "paddleocr_vl"
    # Line crops from a 4000x3000 photo can be 300px+ tall; the model was
    # evaluated (and reads just as well) at this size, with fewer image tokens.
    max_height = 160
    max_width = 1600

    def __init__(
        self,
        model_dir: str,
        device: str | None = None,
        max_new_tokens: int = 128,
        batch_size: int = 1,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = _resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.processor.tokenizer.padding_side = "left"  # batched generation appends on the right
        self.model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=dtype).to(self.device).eval()

    def _prepare(self, crop: np.ndarray) -> Image.Image:
        h, w = crop.shape[:2]
        scale = min(1.0, self.max_height / h, self.max_width / w)
        if scale < 1:
            crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    def recognize(self, crop: np.ndarray) -> str:
        return self.recognize_batch([crop])[0]

    @torch.inference_mode()
    def recognize_batch(self, crops: list[np.ndarray]) -> list[str]:
        texts: list[str] = []
        for start in range(0, len(crops), self.batch_size):
            chunk = crops[start : start + self.batch_size]
            conversations = [
                [{"role": "user", "content": [{"type": "image", "image": self._prepare(crop)}, {"type": "text", "text": "OCR:"}]}]
                for crop in chunk
            ]
            # A line holds about as many characters as it is heights wide, and a
            # token covers ~3 characters; 2 tokens per height plus slack is
            # generous, and stops a runaway loop early instead of at 128 tokens.
            widest = max(crop.shape[1] / max(crop.shape[0], 1) for crop in chunk)
            max_new_tokens = min(self.max_new_tokens, 16 + int(2 * widest))
            inputs = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = self.processor.batch_decode(
                generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            texts.extend(usable(collapse_repeats(" ".join(text.split()))) for text in decoded)
        return texts


SPECIALISTS = {"paddleocr_vl": PaddleOcrVlRecognizer, "got_ocr2": GotOcrRecognizer}


def load_specialist(config: dict):
    """Builds the fallback recognizer named by config["specialist"]["engine"]."""
    engine = config.get("specialist", {}).get("engine", "got_ocr2")
    cfg = config[engine]
    return SPECIALISTS[engine](
        model_dir=cfg["model_dir"],
        device=cfg.get("device"),
        max_new_tokens=cfg.get("max_new_tokens", 128),
        batch_size=cfg.get("batch_size", 1),
    )
