class PaddleDetector:
    def __init__(self, lang: str = "en", use_gpu: bool = False):
        raise NotImplementedError

    def detect(self, image):
        raise NotImplementedError
