r"""
# Models

This module contains the models that are used to predict the properties/performance of sequences based on the measured data and the candidates generated.
A model is used by a policy to select the next batch of sequences to measure in the wetlab.
"""
from .base import SeqPropertyPred, IncrSeqPropertyPred
from .unirep_based import RFuniRep
from .logistic_regr import LogisticRegr
from .kernel_regression_pytorch import (
    KernelRegression,
)

# from .krr import KernelRidgeRegression
from .gpytorch_models import (
    GaussianKernelRegression,
    CustomGPModel,
    GPModelElementwiseKernel,
    GPModelElementwiseNPositionKernel,
)
from .gpytorch_kernels import OneHotIndexKernel, OneHotIndexRBFKernel

from .pretrained import LowerGrowthTemperaturePred, SolubilityPred

# from .gp_unirep import GPuniRep
