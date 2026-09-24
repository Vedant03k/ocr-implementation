import argparse


def main():
    parser = argparse.ArgumentParser(description="OCR pipeline for images and video")
    parser.add_argument("input", help="Path to an image or video file")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default=None, help="Path to write JSON output")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
