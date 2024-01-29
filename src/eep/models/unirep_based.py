from typing import Iterator, Tuple
import datasets as ds
import numpy as np
import jax.numpy as jnp
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import MinMaxScaler
from .base import SeqPropertyPred
from ..util import PRNGKeyT, Array


class RFuniRep(SeqPropertyPred):
    """Random Forest model for sequence property prediction based on UniRep representations."""

    def __init__(
        self, include_variance=False, used_encoding="aa_unirep_1900", **kwargs
    ):
        """Constructor for the RFuniRep model.

        Args:
            include_variance (bool, optional): Whether to include variance in the prediction. Defaults to False.
            See SKlearn.
        """
        super().__init__()
        # self.include_dna = include_dna
        self.include_variance = include_variance
        self.scaler = MinMaxScaler()
        self.rf = RandomForestRegressor(**kwargs)
        self.target_name = None
        self.encoding_name = used_encoding
        print(used_encoding)

    @property
    def features(self) -> Iterator[str]:
        """Returns the features used by the model.

        Yields:
            Feature names
        """
        yield self.encoding_name

    def _extract_x_y(self, ds: ds.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts the features and target values from a dataset.

        Args:
            ds: Dataset to extract features & target values from.

        Returns:
            Features and target values.
        """
        try:
            ds = ds.with_format("jax", columns=[self.encoding_name])
        except BaseException:
            pass
        X = ds[self.encoding_name]
        targ_name, y = SeqPropertyPred.get_target(ds)

        if len(y) == 0:
            return X, None
        else:
            y = np.vstack(y).squeeze()
            return X, y

    def fit(self, rng: PRNGKeyT, train_ds: ds.Dataset):
        """Train the model on a dataset.

        Args:
            rng: Random number generator key.
            train_ds: Training dataset.
        """
        self._store_target_names(train_ds)
        np.random.seed(np.array(rng)[0])
        X, y = self._extract_x_y(train_ds)
        X = self.scaler.fit_transform(X)
        # import pdb
        # pdb.set_trace()
        if self.include_variance:
            if len(y.shape) == 1:
                y = y[:, np.newaxis]
            self.dim_y = y.shape[1]
            y = np.hstack([y, y**2])
        self.rf.fit(X, y)

    def predict(
        self, test_ds: ds.Dataset, flat: bool = False, tolist: bool = False
    ) -> dict[str, Array] | dict[str, dict[str, Array]]:
        """Predict the mean and variance of the target values for a dataset.
        Uses the trained model, i.e. a GP based on the UniRep representation as input.

        Args:
            test_ds: The dataset to predict the target values for.
            flat: Whether to return the mean and variance as a flat dict. Defaults to False.

        Returns:
            Flat or nested dict with mean and variance.
        """
        if isinstance(test_ds, dict):
            test_ds = ds.Dataset.from_dict(test_ds)
        X, _ = self._extract_x_y(test_ds)
        X = self.scaler.transform(X)
        if not self.include_variance:
            m = self.rf.predict(X).squeeze()
            v = np.zeros_like(m)
        else:
            p = self.rf.predict(X).squeeze()
            m, x_squared = p[:, : self.dim_y], p[:, self.dim_y :]
            v = np.clip(x_squared - m**2, 0, None)
        return self._prediction_arrays_to_dictionary(
            jnp.array(m), jnp.array(v), flat=flat, tolist=tolist
        )


class GPuniRep(SeqPropertyPred):
    """GP model for sequence property prediction based on UniRep representations."""

    def __init__(self, **kwargs):
        """Constructor for the GPuniRep model.

        Args:
            See SKlearn GaussianProcessRegressor args.
        """
        super().__init__()
        # self.scaler = MinMaxScaler()
        self.rf = GaussianProcessRegressor(normalize_y=True, **kwargs)

    @property
    def features(self) -> Iterator[str]:
        """Returns the features used by the model.

        Yields:
            Iterator[str]: Feature names
        """
        yield "aa_unirep_1900"

    def _extract_x_y(self, ds: ds.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts the features and target values from a dataset.

        Args:
            ds: Dataset to extract features & target values from.

        Returns:
            Features and target values.
        """

        try:
            ds = ds.with_format("jax", columns=["aa_unirep_1900"])
        except BaseException:
            pass
        X = ds["aa_unirep_1900"]
        targ_name, y = SeqPropertyPred.get_target(ds)

        if len(y) == 0:
            return X, None
        else:
            y = np.vstack(y).squeeze()
            return X, y

    def fit(self, rng: PRNGKeyT, train_ds: ds.Dataset):
        """Train the model on a dataset.

        Args:
            rng: Random number generator key.
            train_ds: Training dataset.
        """
        self._store_target_names(train_ds)
        X, y = self._extract_x_y(train_ds)
        # X = self.scaler.fit_transform(X)
        self.rf.fit(X, y)

    def predict(
        self, test_ds: ds.Dataset, flat: bool = False
    ) -> dict[str, Array] | dict[str, dict[str, Array]]:
        """Predict the mean and variance of the target values for a dataset.
        Uses the trained model, i.e. a GP based on the UniRep representation as input.

        Args:
            test_ds: The dataset to predict the target values for.
            flat: Whether to return the mean and variance as a flat dict. Defaults to False.

        Returns:
            Flat or nested dict with mean and variance.
        """
        X, _ = self._extract_x_y(test_ds)
        #  X = self.scaler.transform(X)

        m, s = self.rf.predict(X, return_std=True)
        return self._prediction_arrays_to_dictionary(
            jnp.array(m.squeeze()), jnp.array(s.squeeze()) ** 2, flat=flat
        )
