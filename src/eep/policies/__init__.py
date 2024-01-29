r"""
# Policies
This module contains the policies that are used to select the next batch of sequences to measure in the wetlab based on the measured data, the candidates generated, and the model predictions.
"""
from .allcandidates import AllCandidates
from .base import BatchOptPolicy
from .indep_ucb import IndepUCB
from .bucb import BatchUCB
from .random import RandomPolicy
from .bthompson import BatchThompson
