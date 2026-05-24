from __future__ import annotations

import platform
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch cuda available: {torch.cuda.is_available()}")
        print(f"torch cuda version: {torch.version.cuda}")
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                print(f"gpu[{idx}]: {torch.cuda.get_device_name(idx)}")
    except Exception as exc:
        print(f"torch import failed: {exc}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                import tilelang

        print(f"tilelang: {tilelang.__version__}")
    except Exception as exc:
        print(f"tilelang import failed: {exc}")


if __name__ == "__main__":
    main()
