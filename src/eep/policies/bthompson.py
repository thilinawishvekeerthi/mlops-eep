from typing import Iterator, Sequence
import datasets as ds
from jax.random import split
import torch
from .base import BatchOptPolicy
from ..util import PRNGKeyT
from ..models import GaussianKernelRegression
import eep.generate as gen


class BatchThompson(BatchOptPolicy):
    def __init__(
        self,
        model: GaussianKernelRegression,
        generator: gen.Generator = gen.RandomGenerator(
            gen.ancestor.BestChoice(), "aa", 1000, 2
        ),
        **param,
    ) -> None:
        """Batch Thompson sampling policy. Samples independently from the posterior distribution of the model and selects the best candidates.

        Args:
            model (GaussianKernelRegression): The model to use for predictions.
            generator (gen.Generator, optional): Candidate generator class. Defaults to gen.RandomGenerator( gen.ancestor.BestChoice(), "aa", 1000, 2).
        """
        super().__init__()
        self.model = model

        self.param = param
        self.generator = generator

    @property
    def features(
        self,
    ) -> Sequence[str] | Iterator[str]:
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
        """Call the batch Thompson sampling policy to select the next batch of sequences.
        Args:
            rng: Random number generator key.
            measured: Measured data in huggingface datasets format.
            batch_size: Size of experimental wetlab batch. Defaults to 96.

        Returns:
            The next batch of data to measure in the wet lab.
        """
        key1, key2 = split(rng, 2)  # FIXME: use the eep function to split the key

        if (
            "dna_seq" in measured.features
        ):  # FIXME - this does not seem to be used. What's the purpose?
            seq_types = ["dna", "aa"]
        else:
            seq_types = ["aa"]

        # Train the model on the measured data
        measured = self.model.add_encodings(measured)
        self.model.fit(key2, measured)

        # Get the index of the target summary variable
        for summary_idx, name in enumerate(self.model.target_name):
            if name == "target summary variable":
                break

        # Sample candidates from the generator
        mutations = self.generator(key1, measured, [])
        df = self.model.add_encodings(ds.Dataset.from_dict({"aa_seq": mutations}))

        # Get the posterior distribution of the model for the candidates
        posterior_mvn = self.model.likelihood(
            self.model.model(self.model.get_features(df)[0])
        )

        selected_indices = []  # FIXME what is this for?

        # more samples than batch_size, since
        # if the argmax is the same between two samples, we can only keep one
        sample = posterior_mvn.sample(torch.Size([batch_size * 5]))[
            :, :, summary_idx
        ].squeeze()
        picked = torch.argmax(sample, axis=1).unique()
        if len(picked) < batch_size:
            raise ValueError("couldn't find enough candidates")
        # FIXME: we should randomize the order of the picked candidates with a random seed,
        # since currently they are ordered by increasing index (through the use of unique())
        # which could have hard to understand interactions with generators
        return df.select(picked[:batch_size])
