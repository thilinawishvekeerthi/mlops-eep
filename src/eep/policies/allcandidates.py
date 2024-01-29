from functools import partial
import datasets as ds
from jax.random import split
from .base import BatchOptPolicy
from ..util import PRNGKeyT
from ..models import IncrSeqPropertyPred
import eep.generate as gen


class AllCandidates(BatchOptPolicy):
    """This policy selects all generated candidates for the next batch to measure.
    Its main use is for returning all predictions to the user, who can then select according to their own criteria.
    """

    def __init__(
        self,
        model: IncrSeqPropertyPred,
        generator: gen.Generator = gen.RandomGenerator(
            gen.ancestor.BestChoice(), "aa", 1000, 2
        ),
        **param,
    ) -> None:
        """Constructor for AllCandidates policy.

        Args:
            model: The model predicting the properties of the sequences.
            generator: The generator to use for generating candidate sequences. Defaults to gen.RandomGenerator(gen.ancestor.BestChoice(), "aa", 1000, 2).
        """
        super().__init__()
        self.model = model

        self.param = param
        self.mutator = generator

    def __call__(
        self,
        rng: PRNGKeyT,
        measured: ds.Dataset,
        batch_size: int = None,
    ) -> ds.Dataset:
        """Call the AllCandidates policy to select the next batch of sequences.
        This ignores the batch_size argument and returns all generated candidates.
        The user can then select the sequences to measure according to their own criteria.

        Args:
            rng: Random number generator key.
            measured: Measured data in huggingface datasets format.
            batch_size: Size of experimental wetlab batch. Parameter is ignored.

        Returns:
            All generated candidates and their predicted properties.
        """
        key1, key2 = split(rng, 2)

        measured = self.model.add_encodings(measured)
        self.model.fit(key2, measured)
        if "dna_seq" in measured.features:
            seq_types = ["dna", "aa"]
        else:
            seq_types = ["aa"]

        mutations = self.mutator(key1, measured, [])

        df = ds.Dataset.from_dict({f"{t}_seq": mutations for t in seq_types})

        df = df.map(
            lambda x: self.model.add_encodings(ds.Dataset.from_dict(x)).to_dict(),
            writer_batch_size=100,
            batch_size=100,
            batched=True,
        )
        rval = df.map(
            partial(self.model.predict, flat=True, tolist=True),
            writer_batch_size=100,
            batch_size=100,
            batched=True,
        )
        return rval
