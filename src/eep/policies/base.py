from abc import ABC, abstractmethod
import datasets as ds
from ..util import PRNGKeyT


class BatchOptPolicy(ABC):
    """Abstract base class for batch optimization policies."""

    @abstractmethod
    def __call__(
        self,
        rng: PRNGKeyT,
        measured: ds.Dataset,
        batch_size: int = 96,
    ) -> ds.Dataset:
        """Call signature for batch optimization policies.
        Args:
            rng: Random number generator key.
            measured: Measured data in huggingface datasets format.
            batch_size: Size of experimental wetlab batch. Defaults to 96.

        Returns:
            The next batch of data to measure in the wetlab.
        """
        pass
