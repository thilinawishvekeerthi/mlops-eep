from typing import Iterator, Sequence, Union
import datasets as ds
from jax.random import split
from .base import BatchOptPolicy
from ..models import IncrSeqPropertyPred
from ..util import PRNGKeyT
import eep.generate as gen


class IndepUCB(BatchOptPolicy):
    """Independent UCB policy. This policy ignores the fact that the measurement process is parallelized/batched over multiple candidates."""

    def __init__(
        self,
        model: IncrSeqPropertyPred,
        exploration_factor: float = 0.5,
        generator: gen.Generator = gen.RandomGenerator(
            gen.ancestor.BestChoice(), "aa", 1000, 2
        ),
        **param,
    ) -> None:
        """Independent UCB policy constructor.

        Args:
            model (IncrSeqPropertyPred): The model to use for predictions.
            exploration_factor (float, optional): The factor for interpolating between exploration and exploitation. Defaults to 0.5.
            generator (gen.Generator, optional): Candidate generator class. Defaults to gen.RandomGenerator( gen.ancestor.BestChoice(), "aa", 1000, 2).
        """
        super().__init__()
        self.model = model

        self.param = param
        self.exploration_factor = exploration_factor
        self.generator = generator

    @property
    def features(
        self,
    ) -> Union[Sequence[str], Iterator[str]]:
        """Get the features of the model.

        Returns:
            Union[Sequence[str], Iterator[str]]: The features of the model.
        """
        return self.model.features

    def __call__(
        self,
        rng: PRNGKeyT,
        measured: ds.Dataset,
        batch_size: int = 96,
    ) -> ds.Dataset:
        """Call the IndepUCB policy to select the next batch of sequences.
        Args:
            rng (PRNGKeyT): Random number generator key.
            measured (ds.Dataset): Measured data in huggingface datasets format.
            batch_size (int, optional): Size of experimental wetlab batch. Defaults to 96.

        Returns:
            ds.Dataset: The next batch of data to measure in the wet lab.
        """
        key1, key2 = split(rng, 2)

        if "dna_seq" in measured.features:
            seq_types = ["dna", "aa"]  # FIXME that is unused
        else:
            seq_types = ["aa"]  # FIXME that is unused

        measured = self.model.add_encodings(measured)
        mutations = self.generator(key1, measured, [])
        self.model.fit(key2, measured)
        print("...training done")
        df = self.model.add_encodings(ds.Dataset.from_dict({"aa_seq": mutations}))

        rval = self.model.preselect(df, batch_size, self.exploration_factor)
        return rval
