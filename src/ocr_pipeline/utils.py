import cv2
import numpy as np


def crop_polygon(image: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Perspective-crops a quadrilateral text region into an upright image."""
    poly = poly.astype(np.float32)
    width = int(max(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3])))
    height = int(max(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2])))
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(poly, dst)
    crop = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    if crop.shape[0] / max(crop.shape[1], 1) >= 1.5:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop
