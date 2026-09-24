class GotOcrRecognizer:
    def __init__(self, model_dir: str, device: str = "cuda"):
        raise NotImplementedError

    def recognize(self, crop):
        raise NotImplementedError
