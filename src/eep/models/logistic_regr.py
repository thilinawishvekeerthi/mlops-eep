"""
Module logistic_regr.py

This module contains the LogisticRegr class, which is a model for sequence classification based on aa_1gram and aa_unirep_1900 features.

Example:
    >>> from eep.models import LogisticRegr
    >>> from pedata.static.example_data.example_data import ClassficationToyDataset
    >>> import tempfile
    >>> dataset = ClassficationToyDataset(["aa_1gram", "aa_unirep_1900"]).train_test_split_dataset
    >>> model = LogisticRegr()
    >>> model.fit(None, dataset['train'])
    >>> pred = model.predict(dataset['test'])
    >>> print(pred['target_solubility'].shape)
    (5, 2)
    >>> fp = tempfile.TemporaryFile()
    >>> model.save_weights(fp)
    >>> fp.seek(0)
    >>> model = LogisticRegr()
    >>> model.load_weights(fp)
    >>> pred = model.predict(dataset['test'])
    >>> print(pred['target_solubility'].shape)
    (5, 2)
    >>> print(pred['target_solubility'])
"""
from os import PathLike
import datasets as ds
from typing import Iterator, Sequence, Tuple, Union
from ..util import PRNGKeyT, Array
import jax.numpy as np
import numpy as onp
from sklearn.linear_model import LogisticRegression
import joblib
from .base import SeqClassifier, SeqPropertyPred
import pedata


class LogisticRegr(SeqClassifier):
    """
    Logistic regression model for sequence classification based on aa_1gram and aa_unirep_1900 features.
    The model predicts the probability for each class. The output array has a shape of (n_samples, n_classes).
    """

    def __init__(self):
        """Initialize the model."""
        super().__init__()

        self.sklearn_model = LogisticRegression()
        self.sklearn_model.target_name = None

    def get_target(
        self, d: ds.Dataset, max_targets: int = None, as_dict: bool = False
    ) -> Tuple[Array]:
        """Check if the target variable is binary.

        Args:
            train_df (ds.Dataset): Dataset to train the model on.

        Returns:
            None
        """
        target_names, y = pedata.util.get_target(d, max_targets, as_dict=as_dict)
        if len(target_names) == 0:
            raise ValueError("No target variable found.")
        elif len(target_names) > 1:
            raise ValueError("Only one target variable allowed.")
        elif len(np.unique(y)) != 2:
            raise ValueError("Target variable must be binary.")
        else:
            return target_names, y

    def fit(self, rng: PRNGKeyT, train_df: ds.Dataset) -> None:
        """Training method for the model.

        Args:
            rng (PRNGKeyT, optional): A random number generator key.
            train_df (ds.Dataset): Dataset to train the model on.

        Returns:
            None
        """
        train_df = self.add_encodings(train_df)
        train_df = train_df.with_format("numpy")
        X = onp.concatenate([train_df[feature] for feature in self.features], axis=1)
        target_names, y = self.get_target(train_df)
        self.sklearn_model.target_name = target_names[0]
        y = y.flatten()
        self.sklearn_model.fit(X, y)

    def predict(self, test_df: ds.Dataset) -> dict[str, Array]:
        """
        Predict the values for the 'test_df' based on the trained model.
        The model predicts the probability for each class. The output array has a shape of (n_samples, n_classes).

        Args:
            test_df (ds.Dataset): Dataset for which to predict target variables.

        Returns:
            dict[str, Array]: A dictionary, where keys are the different target label columns from the dataset. Values are the predicted probabilities for each class. Array has a shape of (n_samples, n_classes).
        """

        test_df = self.add_encodings(test_df)
        test_df = test_df.with_format("numpy")
        X = onp.concatenate([test_df[feature] for feature in self.features], axis=1)
        return {self.sklearn_model.target_name: self.sklearn_model.predict_proba(X)}

    @property
    def features(
        self,
    ) -> Union[Sequence[str], Iterator[str]]:
        """Features, like "aa_1hot", "aa_2gram" etc that can be used by this predictor.

        Raises:
            NotImplementedError: Abstract baseclass method.

        Returns:
            Sequence[str]: The features this predictor would like to use.
        """
        yield from ["aa_1gram", "aa_unirep_1900"]

    def save_weights(self, path: PathLike) -> None:
        """Save the model parameters to a file.

        Args:
            path (PathLike): Path to save the model parameters to.

        Returns:
            None
        """
        joblib.dump(self.sklearn_model, path)

    def load_weights(self, path: PathLike) -> None:
        """Restore the model parameters from a file.

        Args:
            path (PathLike): Path to restore the model parameters from.

        Returns:
            an instance of the class
        """
        self.sklearn_model = joblib.load(path)
