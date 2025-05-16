import os

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    os.environ["TRITON_INTERPRET"] = "1"

torch.set_num_threads(1)


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if HAS_CUDA else "cpu"
