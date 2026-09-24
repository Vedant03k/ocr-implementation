import os as _os

# PaddleX reads PADDLE_PDX_CACHE_HOME once at import time, so it must be set
# before paddleocr/paddlex are imported anywhere in the process. Redirects the
# model cache from ~/.paddlex into the repo for portability (S3/container deploys).
_os.environ.setdefault(
    "PADDLE_PDX_CACHE_HOME",
    _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "models", "paddleocr")),
)

# Importing torch before paddleocr avoids a native DLL load conflict between
# their bundled libraries (observed as `OSError: [WinError 127] ... shm.dll`).
import torch  # noqa: F401,E402
