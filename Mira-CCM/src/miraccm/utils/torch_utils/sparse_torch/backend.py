import importlib
from typing import *

BACKEND = "spconv"


def __from_env():
    import os

    global BACKEND

    env_sparse_backend = os.environ.get("SPARSE_BACKEND")

    if env_sparse_backend is not None and env_sparse_backend in [
        "spconv",
        "torchsparse",
    ]:
        BACKEND = env_sparse_backend

    print(f"[SPARSE] Backend: {BACKEND}")


__from_env()


def set_backend(backend: Literal["spconv", "torchsparse"]):
    global BACKEND
    BACKEND = backend
