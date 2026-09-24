import os as _os
import sys as _sys

# PaddleX reads PADDLE_PDX_CACHE_HOME once at import time, so it must be set
# before paddleocr/paddlex are imported anywhere in the process. Redirects the
# model cache from ~/.paddlex into the repo for portability (S3/container deploys).
_os.environ.setdefault(
    "PADDLE_PDX_CACHE_HOME",
    _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "models", "paddleocr")),
)

# Importing torch before paddleocr avoids a native DLL load conflict between
# their bundled libraries on Windows (observed as `OSError: [WinError 127] ...
# shm.dll`). Linux .so loading doesn't have this conflict, and forcing torch
# in here unconditionally would drag it into containers that only need
# PaddleOCR (deploy/serving/paddleocr_server.py) for no reason.
if _sys.platform == "win32":
    import torch  # noqa: F401,E402
