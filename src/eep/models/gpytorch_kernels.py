from typing import Optional, Union
from gpytorch.lazy import LazyEvaluatedKernelTensor
from linear_operator.operators import LinearOperator
import torch
import gpytorch
from gpytorch.kernels import Kernel, IndexKernel
from torch import Tensor


class OneHotIndexKernel(gpytorch.kernels.Kernel):
    def __init__(self, num_categories=21, **kwargs):
        super(OneHotIndexKernel, self).__init__(**kwargs)
        self.num_categories = num_categories
        self.index_kernel = gpytorch.kernels.IndexKernel(num_tasks=num_categories)

    def one_hot_to_index(self, one_hot_tensor):
        # Convert one-hot tensor to index
        return torch.argmax(one_hot_tensor, dim=-1).squeeze(-1).float()

    def __call__(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> LazyEvaluatedKernelTensor | LinearOperator | Tensor:
        if x2 is None:
            x2 = x1
        return self.forward(
            x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor | LinearOperator:
        # Convert one-hot encoded sequence to indices
        x1_index = self.one_hot_to_index(x1)
        x2_index = self.one_hot_to_index(x2)
        # Compute kernel values using the internal IndexKernel
        kernel_values = self.index_kernel(x1_index, x2_index)
        # Return the kernel values directly
        return kernel_values


class OneHotIndexRBFKernel(gpytorch.kernels.Kernel):
    def __init__(self, num_categories=21, **kwargs):
        super(OneHotIndexRBFKernel, self).__init__(**kwargs)
        self.num_categories = num_categories
        self.index_kernel = gpytorch.kernels.IndexKernel(num_tasks=num_categories)
        self.rbf_kernel = gpytorch.kernels.RBFKernel()

    def one_hot_to_index(self, one_hot_tensor):
        # Convert one-hot tensor to index
        return torch.argmax(one_hot_tensor, dim=-1).squeeze(-1).float()

    def __call__(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> LazyEvaluatedKernelTensor | LinearOperator | Tensor:
        if x2 is None:
            x2 = x1
        return self.forward(
            x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor | LinearOperator:
        # Convert one-hot encoded sequence to indices
        x1_index = self.one_hot_to_index(x1)
        x2_index = self.one_hot_to_index(x2)

        # Compute positional differences for RBF kernel
        x1_positions = torch.arange(x1.size(0)).float()
        x2_positions = torch.arange(x2.size(0)).float()

        # Compute kernel values using the internal IndexKernel
        kernel_values = self.index_kernel(x1_index, x2_index) * self.rbf_kernel(
            x1_positions, x2_positions
        )
        # Return the kernel values directly
        return kernel_values


if __name__ == "__main__":
    pass
