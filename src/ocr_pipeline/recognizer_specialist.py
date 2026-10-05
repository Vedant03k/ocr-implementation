import numpy as np
import torch
from PIL import Image


class GotOcrRecognizer:
    def __init__(
        self,
        model_dir: str,
        device: str | None = None,
        max_new_tokens: int = 128,
        batch_size: int = 4,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device is None or (device.startswith("cuda") and not torch.cuda.is_available()):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
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
