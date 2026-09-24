import argparse
import os

import cv2
import yaml

from .detector import PaddleDetector
from .frame_sampler import sample_frames
from .merge import merge_and_reorder
from .recognizer_specialist import GotOcrRecognizer
from .router import route
from .schema import OCRResult

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(input_path: str, config: dict) -> OCRResult:
    detector = PaddleDetector(lang=config["paddleocr"]["lang"], device=config["paddleocr"].get("device", "cpu"))
    specialist = GotOcrRecognizer(
        model_dir=config["got_ocr2"]["model_dir"],
        device=config["got_ocr2"].get("device"),
    )
    threshold = config["router"]["confidence_threshold"]

    blocks = []
    if os.path.splitext(input_path)[1].lower() in VIDEO_EXTENSIONS:
        sampler_cfg = config["frame_sampler"]
        if sampler_cfg.get("mode", "fps") != "fps":
            raise NotImplementedError("Only frame_sampler.mode: fps is implemented")
        for frame, timestamp in sample_frames(
            input_path,
            fps=sampler_cfg["fps"],
            dedupe_similarity_threshold=sampler_cfg["dedupe_similarity_threshold"],
        ):
            detections = detector.detect(frame)
            routed = route(frame, detections, specialist, threshold)
            blocks.extend(merge_and_reorder(routed, frame_timestamp=timestamp))
    else:
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")
        detections = detector.detect(image)
        routed = route(image, detections, specialist, threshold)
        blocks.extend(merge_and_reorder(routed))

    return OCRResult(blocks=blocks)


def main():
    parser = argparse.ArgumentParser(description="OCR pipeline for images and video")
    parser.add_argument("input", help="Path to an image or video file")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default=None, help="Path to write JSON output")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run(args.input, config)
    output_json = result.model_dump_json(indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
