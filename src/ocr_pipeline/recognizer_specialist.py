import numpy as np
import torch
from PIL import Image


class GotOcrRecognizer:
    def __init__(self, model_dir: str, device: str | None = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device is None or (device.startswith("cuda") and not torch.cuda.is_available()):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(model_dir, device_map=self.device)
        self.processor = AutoProcessor.from_pretrained(model_dir)

    def recognize(self, crop: np.ndarray) -> str:
        image = Image.fromarray(crop[:, :, ::-1])  # BGR -> RGB
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(
            **inputs,
            do_sample=False,
            tokenizer=self.processor.tokenizer,
            stop_strings="<|im_end|>",
            max_new_tokens=1024,
        )
        text = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()
