from functools import partial
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Iterator, Sequence
import os
import pandas as pd
import datasets as ds
import jax.numpy as np
import numpy as onp
import pedata
from ..util import PRNGKeyT, Array


def extract_feat(df: ds.Dataset, obs_prefix: str, feat_name: str) -> Any:
    """Extracts a feature from the dataset.

    Args:
        df: The huggingface dataset.
        obs_prefix The observation prefix, typically "aa" or "dna".
        feat_name: The feature name.

    Returns:
        Any: The feature.
    """
    return df[f"{obs_prefix}_{feat_name}"]


class SeqClassifier(ABC):
    """Abstract base class for sequence classifier models."""

    def __init__(self, label=None) -> None:
        """Abstract base class for sequence property prediction models."""
        super().__init__()
        self.target_name = label

    @abstractmethod
    def fit(self, rng: PRNGKeyT, train_df: ds.Dataset) -> None:
        """Training method for the model.

        Args:
            rng: A random number generator key.
            train_df: Dataset to train the model on.

        Raises:
            NotImplementedError: Abstract baseclass method.

        Returns:
            None
        """

        raise NotImplementedError

    @abstractmethod
    def predict(self, test_df: ds.Dataset) -> dict[str, Array]:
        """Predict the values for the 'test_df' based on the trained model.

        Args:
            test_df (ds.Dataset): Dataset for which to predict target variables.

        Returns:
            dict[str, Array]: A dictionary, where keys are the different target label columns from the dataset. Values are the predicted probability of the target label being 1.
        """

        raise NotImplementedError

    @abstractmethod
    def save_weights(self, path: os.PathLike) -> None:
        """Save the model parameters to a file.

        Args:
            path (os.PathLike): The path to save the model parameters to.

        Raises:
            NotImplementedError: Abstract baseclass method.
        """

        raise NotImplementedError

    @abstractmethod
    def load_weights(self, path: os.PathLike) -> None:
        """Restore the model parameters from a file.

        Args:
            path (os.PathLike): The path to restore the model parameters to.

        Raises:
            NotImplementedError: Abstract baseclass method.
        """

        raise NotImplementedError

    @abstractproperty
    def features(
        self,
    ) -> Sequence[str] | Iterator[str]:
        """Features, like "aa_1hot", "aa_2gram" etc that can be used by this predictor.

        Raises:
            NotImplementedError: Abstract baseclass method.

        Returns:
            The features this predictor would like to use.
        """
        raise NotImplementedError

    def add_encodings(
        self,
        measured: ds.Dataset,
    ) -> ds.Dataset:
        """Add encodings to the dataset based on the features the prediction model needs.

        Args:
            measured: The dataset to add encodings/features to.

        Returns:
            The dataset with added encodings/features.
        """
        f = list(self.features)
        return pedata.config.encoding_specs.add_encodings(measured, f)


class SeqPropertyPred(ABC):
    """Abstract base class for sequence property prediction models."""

    def __init__(self) -> None:
        """Abstract base class for sequence property prediction models."""
        super().__init__()
        self.target_name = None

    def _store_target_names(self, train_ds: ds.Dataset):
        """Store the target names from the dataset for later usage

        Args:
            train_ds (ds.Dataset): The training dataset in huggingface format.
        """
        self.target_name, _ = SeqPropertyPred.get_target(train_ds)

    def _prediction_arrays_to_dictionary(
        self,
        mean: Array = None,
        variance: Array = None,
        flat: bool = False,
        tolist: bool = False,
    ) -> dict[str, Array] | dict[str, dict[str, Array]]:
        """Put mean and variance return value into a dictionary structure.

        Args:
            mean (Array): Mean prediction. If None, will not be included in the dictionary.
            variance (Array): Variance prediction. If None, will not be included in the dictionary.
            flat (bool, optional): If True, will return a dictionary with keys "{target name} mean" and "{target name} variance". If False, will return a nested dictionary with first level keys bein "{target name}" and second level "mean"/"var". Defaults to False.
            tolist (bool, optional): If True, will convert the arrays to lists. Defaults to False.

        Returns:
            Union[dict[str, Array], dict[str, dict[str, Array]]]: If flat is False: a dictionary, where keys are the different target columns from the dataset, and values are dictionaries with "mean" and "var" keys for the predictions. If flat is True, will return a dictionary with keys "{target name} mean"/"{target name} var"
        """
        # No huggingface dataset input because it makes this undifferentiable
        # by Jax.

        if tolist:
            cast_func = lambda x: x.tolist()  # FIX ME - make this look better
        else:
            cast_func = lambda x: x

        if mean is not None and len(mean.shape) == 1:
            mean = mean[:, np.newaxis]

        if mean is not None and variance is not None:
            if len(variance.shape) == 1:
                variance = variance[:, np.newaxis]

        rval = {}
        for i, name in enumerate(self.target_name):
            if flat:
                if mean is not None:
                    rval[f"{name} mean"] = cast_func(mean.T[i])
                if variance is not None:
                    rval[f"{name} var"] = cast_func(variance.T[i])
            else:
                rval[name] = {}
                if mean is not None:
                    rval[name]["mean"] = cast_func(mean.T[i])
                if variance is not None:
                    rval[name]["var"] = cast_func(variance.T[i])
        return rval

    @classmethod
    def get_target(
        cls, d: ds.Dataset, max_targets: int = None, as_dict: bool = False
    ) -> (
        dict[Sequence[str], onp.ndarray] | tuple[Sequence[str], onp.ndarray]
    ):  # FIXME not sure if this is the right type onp.ndarray
        """Extract target variables from dataset.

        Args:
            d: Dataset to extraxt target variables from

        Returns:
            The target variable name(s) and values
        """
        return pedata.util.get_target(d, max_targets, as_dict=as_dict)

    @abstractmethod
    def fit(self, rng: PRNGKeyT, train_df: ds.Dataset) -> None:
        """Training method for the model.

        Args:
            rng: A random number generator key.
            train_df: Dataset to train the model on.

        Raises:
            NotImplementedError: Abstract baseclass method.

        Returns:
            None
        """

        raise NotImplementedError

    @abstractmethod
    def predict(
        self, test_df: ds.Dataset | dict[str, Array]
    ) -> dict[str, dict[str, Array]]:
        """Predict the values for the 'test_df' based on the trained model.

        Args:
            test_df: Dataset for which to predict target variables.

        Returns:
            A dictionary, where keys are the different target columns from the dataset.
            Values are dictionaries again with "mean" and "var" keys and the respective prediction as values.
        """

        raise NotImplementedError

    def ucb_criterion(self, test_df: ds.Dataset, exploration_factor: float) -> Array:
        """Compute the upper confidence bound criterion for the test sequences given in `test_df`.

        Args:
            test_df: The test sequences given in huggingface dataset format.
            exploration_factor: The factor to interpolate between exploration and exploitation.

        Returns:
            Array: The upper confidence bound criterion.
        """
        pred = test_df.with_format("numpy").map(
            partial(self.predict, flat=True),
            writer_batch_size=100,
            batch_size=100,
            batched=True,
        )
        ucb = pred["target summary variable mean"] + exploration_factor * np.sqrt(
            np.clip(pred["target summary variable var"], 0)
        )
        return ucb

    def preselect(
        self,
        test_df: ds.Dataset,
        n: int,
        exploration_factor: float,
    ) -> ds.Dataset:
        """Select `n` sequences from `test_df` with the highest upper confidence bound (UCB) criterion.

        Args:
            test_df: The test sequences given in huggingface dataset format.
            n: The number of sequences to select.
            exploration_factor: The factor to interpolate between exploration and exploitation.

        Returns:
            The selected sequences (sorted by descending UCB) and predictions.
        """
        test_df = test_df.with_format("numpy").map(
            partial(self.predict, flat=True),
            writer_batch_size=100,
            batch_size=100,
            batched=True,
        )
        ucb = test_df["target summary variable mean"] + exploration_factor * np.sqrt(
            np.clip(test_df["target summary variable var"], 0)
        )
        selection = list(np.argsort(-ucb))
        if n is not None:
            selection = selection[:n]
        return test_df.select(selection)

    @abstractproperty
    def features(
        self,
    ) -> Sequence[str] | Iterator[str]:
        """Features, like "aa_1hot", "aa_2gram" etc that can be used by this predictor.

        Raises:
            NotImplementedError: Abstract baseclass method.

        Returns:
            The features this predictor would like to use.
        """
        raise NotImplementedError

    def add_encodings(
        self,
        measured: ds.Dataset,
    ) -> ds.Dataset:
        """Add encodings to the dataset based on the features the prediction model needs.

        Args:
            measured: The dataset to add encodings/features to.

        Returns:
            The dataset with added encodings/features.
        """
        f = list(self.features)
        return pedata.config.encoding_specs.add_encodings(measured, f)


class PreTrained(ABC):
    """Abstract base class for pretrained models."""

    def __init__(self) -> None:
        """Abstract base class for sequence property prediction models."""
        super().__init__()

    @abstractproperty
    def trained_dataset(
        self,
    ) -> str:
        """The name of the dataset the predictor has been trained on.

        Raises:
            NotImplementedError: Abstract baseclass method.

        Returns:
            str: The name of the dataset the predictor has been trained on.
        """
        raise NotImplementedError
    
    @abstractproperty
    def trained_split(
        self,
    ) -> str:
        """The name of the dataset split the predictor has been trained on.

        Raises:
            NotImplementedError: Abstract baseclass method.

        Returns:
            str: The name of the dataset split the predictor has been trained on.
        """
        raise NotImplementedError

    @abstractproperty
    def target_name(
        self,
    ) -> Sequence[str] | Iterator[str]:
        """Names of the targets the predictor has been trained on.

        Raises:
            NotImplementedError: Abstract baseclass method.

        Returns:
            Sequence[str]: The targets this predictor would like to use.
        """
        raise NotImplementedError

    @abstractproperty
    def features(
        self,
    ) -> Sequence[str] | Iterator[str]:
        """Names of the features the predictor has been trained on.

        Raises:
            NotImplementedError: Abstract baseclass method.

        Returns:
            Sequence[str]: The features this predictor would like to use.
        """
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, path: os.PathLike) -> None:
        """Load the model parameters from a file or cloud storage.

        Args:
            path (os.PathLike): The path to restore the model parameters to.

        Raises:
            NotImplementedError: Abstract baseclass method.
        """

        raise NotImplementedError

    @abstractmethod
    def predict(
        self, test_df: ds.Dataset | dict[str, Array]
    ) -> dict[str, dict[str, Array]]:
        """Predict the values for the 'test_df' based on the trained model.

        Args:
            test_df (ds.Dataset): Dataset for which to predict target variables.

        Returns:
            dict[str, dict[str, Array]]: A dictionary, where keys are the different target columns from the dataset. Values are dictionaries again with "mean" and "var" keys and the respective prediction as values.
        """

        raise NotImplementedError


class IncrSeqPropertyPred(SeqPropertyPred):
    """Abstract base class for sequence property prediction models that can be updated incrementally."""

    @abstractmethod
    def set_candidates(
        self,
        cand_ds: ds.Dataset,
        max_candidates: int = None,
        exploration_factor: float = None,
    ):
        """Abstract base class for incremental (that is: updatable) sequence property predictors.

        Args:
            cand_ds: Candidates to choose from.
            max_candidates: How many from the top candidates to include in update procedure. Defaults to None, in which case all are used.
            exploration_factor: The factor to interpolate between exploration and exploitation. Defaults to None.

        Raises:
            NotImplementedError: This is an abstract base class.
        """

        raise NotImplementedError

    @abstractmethod
    def pick_candidate(self, idx: int):
        """Pick the candidate with index `idx` from the candidates set.

        Args:
            idx: The index of the candidate to pick.

        Raises:
            NotImplementedError: Abstract base class method.

        """

        raise NotImplementedError

    @abstractmethod
    def remaining(self) -> pd.DataFrame:
        """Return the remaining candidates.

        Raises:
            NotImplementedError: Abstract base class method.

        Returns:
            The remaining candidates with their respective updated upper confidence bound criterion.
        """

        raise NotImplementedError
