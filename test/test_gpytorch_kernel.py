import torch
from eep.models import OneHotIndexKernel, OneHotIndexRBFKernel
import pytest


@pytest.fixture()
def x1():
    return torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )


@pytest.fixture()
def x2():
    return torch.tensor(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )


def test_OneHotIndexRBFKernel(x1, x2):
    kernel = OneHotIndexRBFKernel(num_categories=3)
    kernel_matrix = kernel(x1, x2)
    print(kernel_matrix.to_dense())


def test_OneHotIndexKernel(x1, x2):
    kernel = OneHotIndexKernel(num_categories=3)
    kernel_matrix = kernel(x1, x2)
    print(kernel_matrix.to_dense())
