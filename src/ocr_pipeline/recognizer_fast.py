class PaddleRecognizer:
    def __init__(self, lang: str = "en", use_gpu: bool = False):
        raise NotImplementedError

    def recognize(self, crop):
        raise NotImplementedError
