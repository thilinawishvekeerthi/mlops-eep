from abc import ABC, abstractmethod
import datasets as ds
from jax.random import PRNGKey, split
from pedata.config.encoding_specs import add_encodings
from ..models.base import SeqPropertyPred
from .typing import PRNGKeyT


class ArtificialFitness(ABC):
    """Abstract base class for artificial fitness functions."""

    def ensure_ds(self, candidates: ds.Dataset | list[str]) -> ds.Dataset:
        """Computes 'aa' encoding if 'candidates' is a Dataset.

        Args:
            candidates: data which should be encoded

        Returns:
            Data from candidates and encoding as dataset
        """
        if not isinstance(candidates, ds.Dataset):
            candidates = ds.Dataset.from_dict(
                {f"{self.candidate_seq_type}_seq": candidates}
            )

        if self.candidate_seq_type == "dna" and "aa_seq" not in candidates:
            candidates = add_encodings(candidates, ["aa_seq"])
        return candidates

    @abstractmethod
    def __call__(self, candidates: ds.Dataset | list[str]) -> ds.Dataset:
        """call function which is overwritten by *AF

        Args:
            candidates: data which should be encoded

        Returns:
            a Dataset
        """
        pass


class PredictiveModelAF(ArtificialFitness):
    """Artificial fitness function based on a predictive model."""

    def __init__(
        self,
        df: ds.Dataset,
        model: SeqPropertyPred,
        candidate_seq_type: str,
        num_starting_variants: int | float = 0.1,
        rng: PRNGKeyT = PRNGKey(0),
        exploration: float = 0.0,
        **param,
    ):
        """Computes 'aa' encoding if 'candidates' is a Dataset.

        Args:
            df: Dataset for training
            model: model which is fitted to the data
            num_starting_variants: Absolute number of starting variants if passing and integer. Otherwise, fraction of variants in training set to use as starting variants. Defaults to 0.1.
            rng: random number used for training (default is PRNGKey(0))
            **param: parameters for model training
        """

        default_params = {
            "opt_steps": 10,
            "pos": ["Gaussian", 0.4, 1.0],
            "out": ["Laplace", 0.4],
            "regul": None,
            "ndata": 2000,
        }

        for k in default_params:
            if k not in param:
                param[k] = default_params[k]
        # get rand key
        key1, key2 = split(rng, 2)

        self.model = model
        self.candidate_seq_type = candidate_seq_type

        train = df
        assert (
            isinstance(num_starting_variants, float)
            and num_starting_variants > 0
            and num_starting_variants <= 1
        ) or (
            isinstance(num_starting_variants, int)
            and num_starting_variants >= 1
            and num_starting_variants <= len(train)
        ), f"For num_starting_variants got {num_starting_variants}, expected a float between 0 and 1 or an integer between 1 and {len(train)}"

        if isinstance(num_starting_variants, float):
            num_starting_variants = max(int(len(train) * num_starting_variants), 1)

        self.starting_variants = train.sort("target summary variable")[
            :num_starting_variants
        ]["aa_seq"]
        self.optimal = train.sort("target summary variable", reverse=True)[
            :10
        ]  # .drop(columns=["index",])
        self.exploration = exploration
        self.model.fit(rng, train)

    def __call__(self, candidates: ds.Dataset | list[str]) -> ds.Dataset:
        """Uses the predictive model as a wet-lab standin, "measuring" fitness for the given set of `candidates`.

        Args:
            candidates: data which should be encoded

        Returns:
            ds.Dataset
        """
        rval = self.model.add_encodings(self.ensure_ds(candidates))

        for i, t_n in enumerate(self.model.target_name):
            rval = rval.with_format("numpy")
            rval = rval.add_column(
                t_n, self.model.ucb_criterion(rval, self.exploration).tolist()
            )
        return rval


class CountAAcidAF(ArtificialFitness):
    """Abstract base class for artificial fitness functions."""

    def __init__(self, idx: int, candidate_seq_type: str = "aa"):
        self.idx = idx
        self.candidate_seq_type = candidate_seq_type

    def ensure_ds(self, candidates: ds.Dataset | list[str]) -> ds.Dataset:
        """Computes 'aa' encoding if 'candidates' is a Dataset.

        Args:
            candidates: data which should be encoded

        Returns:
            data from candidates and encoded as dataset
        """
        if not isinstance(candidates, ds.Dataset):
            candidates = ds.Dataset.from_dict(
                {f"{self.candidate_seq_type}_seq": candidates}
            )
        return add_encodings(candidates, ["aa_1hot", "aa_len"])

    def __call__(self, candidates: ds.Dataset | list[str]) -> ds.Dataset:
        """Count amino acids with index self.idx

        Args:
            candidates: data which should be encoded

        Returns:
            ds.Dataset
        """
        c = self.ensure_ds(candidates)
        rval = c.add_column(
            "target summary variable",
            c.with_format("numpy")["aa_1hot"]
            .reshape(len(c), -1, 21)[:, :, self.idx]
            .sum(1)
            .squeeze()
            .tolist(),
        )
        return rval
