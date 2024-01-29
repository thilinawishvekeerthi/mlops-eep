import datasets as ds
from jax.random import split, permutation
from .base import BatchOptPolicy
from ..util import PRNGKeyT
import eep.generate as gen
import eep.generate.ancestor as anc


class RandomPolicy(BatchOptPolicy):
    """Random policy. This policy selects a random batch of sequences from the candidate pool.
    In conjunction with a `RandomGenerator`, this policy is equivalent to a random search, aka directed evolution.
    """

    def __init__(
        self,
        mutator: gen.Generator = gen.RandomGenerator(anc.BestChoice(), "aa", 1000, 2),
    ) -> None:
        """Constructor for the RandomPolicy.

        Args:
            mutator (gen.Generator, optional): Candidate generator class. Defaults to gen.RandomGenerator(anc.BestChoice(), "aa", 1000, 2).
        """
        super().__init__()
        self.mutator = mutator

    def __call__(
        self,
        rng: PRNGKeyT,
        measured: ds.Dataset,
        batch_size: int = 96,
    ) -> ds.Dataset:
        """Call the RandomPolicy to select the next batch of sequences.
        Args:
            rng (PRNGKeyT): Random number generator key.
            measured (ds.Dataset): Measured data in huggingface datasets format.
            batch_size (int, optional): Size of experimental wetlab batch. Defaults to 96.

        Returns:
            ds.Dataset: The next batch of data to measure in the wet lab.
        """
        key1, key2 = split(rng)
        if "dna_seq" in measured.features:
            seq_types = ["dna", "aa"]
        else:
            seq_types = ["aa"]
        mut_type = seq_types[0]

        mutations = self.mutator(key1, measured, [])

        # .shuffle(int(key2[0]))
        # if len(mutations) <= batch_size
        return ds.Dataset.from_dict(
            {
                f"{t}_seq": mutations[
                    permutation(key2, len(mutations))[:batch_size].tolist()
                ]
                for t in seq_types
            }
        )
