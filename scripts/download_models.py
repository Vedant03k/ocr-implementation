"""Downloads model weights. PaddleOCR caches to ~/.paddlex/official_models
(its 3.x/PaddleX backend does not support downloading into a custom dir);
GOT-OCR2.0 downloads into models/got_ocr2."""
import os

from huggingface_hub import snapshot_download
from paddleocr import PaddleOCR

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def download_paddleocr():
    PaddleOCR(lang="en")


def download_got_ocr2():
    snapshot_download(
        "stepfun-ai/GOT-OCR-2.0-hf",
        local_dir=os.path.join(MODELS_DIR, "got_ocr2"),
    )


def download_llm_cleanup():
    snapshot_download(
        "Qwen/Qwen2.5-1.5B-Instruct",
        local_dir=os.path.join(MODELS_DIR, "llm_cleanup"),
    )


if __name__ == "__main__":
    print("Downloading PaddleOCR weights...")
    download_paddleocr()
    print("Downloading GOT-OCR2.0 weights...")
    download_got_ocr2()
    print("Downloading LLM cleanup weights...")
    download_llm_cleanup()
    print("Done.")
