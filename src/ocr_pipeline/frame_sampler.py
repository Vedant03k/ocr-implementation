import cv2
import numpy as np


def sample_frames(video_path: str, fps: float = 1.0, dedupe_similarity_threshold: float = 0.97):
    """Yields (frame, timestamp_seconds) at roughly `fps`, skipping near-duplicate frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frame_interval = max(int(round(source_fps / fps)), 1)

    prev_gray = None
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is None or _similarity(prev_gray, gray) < dedupe_similarity_threshold:
                    prev_gray = gray
                    yield frame, frame_index / source_fps
            frame_index += 1
    finally:
        cap.release()


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = cv2.resize(a, (64, 64)).astype(np.float32)
    b = cv2.resize(b, (64, 64)).astype(np.float32)
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-6
    return float(np.sum(a * b) / denominator)
