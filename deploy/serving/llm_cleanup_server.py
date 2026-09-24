"""KServe custom predictor for the LLM cleanup stage (Qwen2.5-1.5B-Instruct), CPU-only.

Custom container instead of KServe's built-in HuggingFace ServingRuntime with
the vLLM backend: vLLM has no practical CPU inference path, and this cluster
currently has no GPU capacity (see deploy/README.md "Current deployment
status"). Wraps ocr_pipeline.cleanup.TextCleaner directly, reusing the exact
prompt/parsing logic the local dev pipeline uses, via plain
transformers.generate() on CPU.

Each instance is one document's ordered lines, cleaned together in one call
so the model can use neighboring lines as context (see TextCleaner's
docstring). Batching multiple documents into one request is supported by the
protocol (one instance per document) even though the orchestrator currently
sends one instance per request.

Request:  {"instances": [{"lines": ["line one", "line two", ...]}]}
Response: {"predictions": [["cleaned line one", "cleaned line two", ...]]}
"""

import os

from kserve import Model, ModelServer

from ocr_pipeline.cleanup import TextCleaner


class LlmCleanupModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.cleaner: TextCleaner | None = None
        self.ready = False

    def load(self):
        self.cleaner = TextCleaner(
            model_dir=os.environ.get("LLM_CLEANUP_MODEL_DIR", "/model"),
            device=os.environ.get("LLM_CLEANUP_DEVICE", "cpu"),
        )
        self.ready = True

    def predict(self, payload: dict, headers: dict | None = None) -> dict:
        predictions = [self.cleaner.clean_lines(instance["lines"]) for instance in payload["instances"]]
        return {"predictions": predictions}


if __name__ == "__main__":
    model = LlmCleanupModel(os.environ.get("MODEL_NAME", "llm-cleanup"))
    model.load()
    ModelServer().start([model])
