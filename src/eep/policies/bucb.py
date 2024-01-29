import datasets as ds
from jax.random import split
from .base import BatchOptPolicy
from ..models import IncrSeqPropertyPred
from ..util import PRNGKeyT
import eep.generate as gen


class BatchUCB(BatchOptPolicy):
    """Batch Upper Confidence Bound (UCB) policy.This policy is a batch version of the IndepUCB policy.
    It is designed to avoid selecting too similar sequences for the next batch, while
    still having a good exploration-exploitation trade-off. (No regret is guaranteed).
    See Desautels, Krause and Burdick, 2012 (Parallelizing exploration-exploitation tradeoffs with Gaussian process bandit optimization).
    """

    def __init__(
        self,
        model: IncrSeqPropertyPred,
        exploration_factor: float = 0.5,
        generator: gen.Generator = gen.RandomGenerator(
            gen.ancestor.BestChoice(), "aa", 1000, 2
        ),
        max_candidates: int = None,
        **param,
    ) -> None:
        """Batch UCB policy constructor.

        Args:
            model (IncrSeqPropertyPred): The model to use for predictions.
            exploration_factor (float, optional): The factor for interpolating between exploration and exploitation. Defaults to 0.5.
            generator (gen.Generator, optional): Candidate generator class. Defaults to gen.RandomGenerator( gen.ancestor.BestChoice(), "aa", 1000, 2 ).
            max_candidates (int, optional): Maximum number of candidates to consider for assembling the batch. After the first prediction, only consider the top `max_candidates` number of candidates. Defaults to None, in which case no cutoff is done.
        """
        super().__init__()
        self.model = model

        self.param = param
        self.exploration_factor = exploration_factor
        self.mutator = generator
        self.max_candidates = max_candidates

    def __call__(
        self,
        rng: PRNGKeyT,
        measured: ds.Dataset,
        batch_size: int = 96,
    ) -> ds.Dataset:
        """Call the BatchUCB policy to select the next batch of sequences.
        Args:
            rng (PRNGKeyT): Random number generator key.
            measured (ds.Dataset): Measured data in huggingface datasets format.
            batch_size (int, optional): Size of experimental wetlab batch. Defaults to 96.

        Returns:
            ds.Dataset: The next batch of data to measure in the wet lab.
        """
        key1, key2 = split(rng, 2)  # FIXME use the eep function to split the key

        self.model.fit(key2, measured)

        if "dna_seq" in measured.features:
            seq_types = ["dna", "aa"]
        else:
            seq_types = ["aa"]

        mutations = self.mutator(key1, measured, [])
        print(f"{len(mutations)} mutations generated")
        df = self.model.add_encodings(
            ds.Dataset.from_dict({f"{t}_seq": mutations for t in seq_types})
        )

        df = self.model.preselect(df, self.max_candidates, self.exploration_factor)
        self.model.set_candidates(
            df,
            exploration_factor=self.exploration_factor,
        )
        if len(df) <= batch_size:
            return df
        else:
            for i in range(batch_size):
                to_pick = self.model.remaining()
                to_pick["ucb"] = (
                    to_pick.pred_mean + self.exploration_factor * to_pick.pred_sd
                )
                pick_idx = int(
                    to_pick.sort_values("ucb", ascending=False)
                    .reset_index()
                    .drop(columns=["index", "ucb"])
                    .iloc[0]
                    .cand_idx
                )
                self.model.pick_candidate(pick_idx)

            return df.select(self.model.picked)
