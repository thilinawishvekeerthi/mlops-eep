"""
Module gpytorch_models.py
This module implements Gaussian Process models using Gpytorch.
The models are trained using gradient based optimization of the kernel parameters.

The main class is GaussianKernelRegression, which can implement a gaussian kernel regression using a a chosen dataset and a chosen GP model.
The GP model can be chosen from the following Model Classes:
- CustomGPModel: A basic GP model with a constant mean and a RBF x linear kernel.
- GPModelElementwiseKernel: A GP model using a constant mean and elementwise sequence comparison.
- GPModelElementwiseNPositionKernel: A GP model using a constant mean and elementwise sequence comparison with a RBF kernel.

Example usage:
----------------
>>> from eep.models import GaussianKernelRegression, CustomGPModel
>>> from datasets import load_dataset
>>> dataset_name = "Company/CrHydA1_PE_REGR"
>>> dataset = load_dataset(dataset_name)["train"].train_test_split(0.2, seed=42)
>>> feature_name = ["aa_1hot"]

>>> # CustomGPModel
>>> model = GaussianKernelRegression(["aa_1hot"], target_standardization=True,epochs=50, modelClass=CustomGPModel)
>>> model.fit(rng=0, train_ds=dataset["train"])
>>> test_prediction = model.predict(test_ds=dataset["test"])

>>> # GPModelElementwiseKernel
>>> model = GaussianKernelRegression(["aa_1hot"], target_standardization=True, epochs=50, modelClass=GPModelElementwiseKernel)
>>> model.fit(rng=0, train_ds=dataset["train"])
>>> test_prediction = model.predict(test_ds=dataset["test"])

>>> # GPModelElementwiseNPositionKernel 
>>> model = GaussianKernelRegression(["aa_1hot"], target_standardization=True, epochs=50, modelClass=GPModelElementwiseNPositionKernel)
>>> model.fit(rng=0, train_ds=dataset["train"])
>>> test_prediction = model.predict(test_ds=dataset["test"])

----------------
"""
from typing import Iterator
import datasets as ds
import numpy as np
from jax.random import PRNGKey
import torch
from torch.optim import Adam
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from pedata.util import (
    DatasetHandler,
    zscore,
    de_zscore_predictions,
)
from ..util.typing import PRNGKeyT
from .base import SeqPropertyPred
from .gpytorch_kernels import OneHotIndexKernel, OneHotIndexRBFKernel


### ======== REGRESSION Class  ============
class GaussianKernelRegression(SeqPropertyPred):
    """
    GyPytorch model with the possibilitiy to use gradient based fitting of kernel parameters
    or use fixed kernels and the closed form fit of the GP.
    """

    def __init__(
        self,
        feature_list: list[str],
        epochs: int = 50,
        target_standardization: str = True,
        modelClass: gpytorch.models = None,
    ):
        """Initialize the regressor

        Args:
            feature_list: Features to use. Defaults to None.
            epochs: Number of epochs to train the kernel. Defaults to 50.
            target_standardization: If True, will standardize the target values (substract the mean and divide by standard deviation).
                Default to True.
            modelClass: The GP model class to use.
                Defaults to CustomGPModel.
                | CustomGPModel (default): A basic GP model with a constant mean and a RBF x linear kernel.
                | GPModelElementwiseKernel: A GP model using a constant mean and elementwise sequence comparison.
                | GPModelElementwiseNPositionKernel: A GP model using a constant mean and elementwise sequence comparison with a RBF kernel.

        Raises:
            ValueError: If feature_list is None.

        """
        super().__init__()
        if feature_list is None:
            raise ValueError("feature_list must be specified")
        self.feat_list = feature_list
        if modelClass is None:
            modelClass = CustomGPModel
        self.ModelClass = modelClass
        self.target_standardization = target_standardization
        self.epochs = epochs

    @property
    def features(self) -> Iterator[str]:
        """Return the names of the features used by the model.

        Yields:
            The feature names.
        """
        yield from self.feat_list

    def get_features(self, dataset: ds.Dataset) -> torch.Tensor:
        """Get the features from the dataset.

        Args:
            dataset: Get the features defined in self.feat_list from Dataset d.

        Returns:
            The features, shape (n_samples, d_features).
        """
        # get all features in self.feat_list from the dataset, reshape to (n_samples, -1)
        # then concatenate them to a single array
        md_features = DatasetHandler(dataset, self.feat_list)
        return md_features.cat(dataset).to(torch.float32), md_features

    def fit(self, rng: PRNGKeyT, train_ds: ds.Dataset) -> None:
        """Train the model on the given dataset.

        Args:
            rng: Random number generator key.
            train_ds: Training dataset.
        """
        # set the random seed
        torch.manual_seed(rng[0])

        # get the target values
        self._store_target_names(train_ds)

        # get the target values and standardize them
        _, target_with_nans = self.get_target(train_ds, as_dict=False)

        if self.target_standardization:
            self.train_target_mean = np.nanmean(target_with_nans, axis=0, keepdims=True)
            self.train_target_std = np.nanstd(target_with_nans, axis=0, keepdims=True)

            target_with_nans = zscore(target_with_nans)

        target_with_nans = torch.asarray(target_with_nans).float()
        self.train_target = target_with_nans
        self.best_train_pred_mean = target_with_nans.max()

        # get the features
        feat, md_features = self.get_features(train_ds)

        # train the model
        self.trainer = KernelTrainer(
            self.ModelClass, feat, target_with_nans, md_features, epochs=self.epochs
        )
        self.model, self.likelihood = self.trainer.fit()

    def predict(
        self,
        test_ds: ds.Dataset | dict[str, np.ndarray],
        flat: bool = False,
        tolist: bool = False,
    ) -> dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]]:
        """Predict the mean and variance of the GP model for the given test dataset.

        Args:
            test_ds: Test dataset.
            flat: If True, will return a dictionary with keys "{target name} mean" and "{target name} variance". If False, will return a nested dictionary with first level keys bein "{target name}" and second level "mean"/"var". Defaults to False.
            tolist: If True, will convert the arrays to lists. Defaults to False.

        Returns:
            Dictionmary with the predicted mean and variance for each target.
        """
        feat, _ = self.get_features(test_ds)
        pred_mean, pred_var = gaussian_predict(self.model, self.likelihood, feat)

        if self.target_standardization:
            # de-zscore

            pred_mean, pred_std = de_zscore_predictions(
                zs_pred_means=pred_mean,
                zs_pred_stds=np.sqrt(pred_var),
                mean=self.train_target_mean,
                std=self.train_target_std,
            )
            pred_var = pred_std**2

        return self._prediction_arrays_to_dictionary(
            pred_mean, pred_var, flat=flat, tolist=tolist
        )


# ========= KERNEL Trainer ==========
class KernelTrainer:
    """Trainer class for the GP model.
    Args:
        ModelClass: The GP model class to use.
        X: The training features.
        y: The training targets.
        md: The DatasetHandler for the training dataset.
        likelihood: The likelihood to use. Defaults to GaussianLikelihood.
        learning_rate: The learning rate for the optimizer. Defaults to 0.1.
        epochs: The number of epochs to train the model. Defaults to 50.

    Returns:
        model: The trained GP model.
        likelihood: The trained likelihood.
    """

    def __init__(
        self,
        ModelClass,
        X: torch.Tensor,
        y: torch.Tensor,
        md: DatasetHandler,
        likelihood=None,
        learning_rate: float = 0.1,
        epochs: int = 50,
    ) -> None:
        """Initialize the model and likelihood, set the training data, define the loss and optimizer"""

        not_nan = ~torch.isnan(y).all(axis=1)
        self.X = X[not_nan, :].to(torch.float32)
        self.y = y[not_nan, :].to(torch.float32)

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, but have {self.X.shape[0]} and {self.y.shape[0]} respectively."
            )

        if likelihood is None:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=y.shape[1]
            )
        else:
            self.likelihood = likelihood

        self.model = ModelClass(X, y, self.likelihood, md)  # prior
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.learning_rate = learning_rate
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs

    def fit(
        self, verbose: bool = False
    ) -> tuple:  # FIXME: fix the type - maybe define types for the type of models
        """Train the model on the dataset."""
        self.model.train()
        self.likelihood.train()

        for i in range(self.epochs):
            self.optimizer.zero_grad()
            self.y_ = self.model(self.X)
            self.loss = -self.mll(self.y_, self.y)
            self.loss.backward()
            if verbose:
                print(
                    "Iter %d/%d - Loss: %.3f" % (i + 1, self.epochs, self.loss.item())
                )
            self.optimizer.step()

        # Set the model and likelihood to evaluation mode
        self.model.eval()
        self.likelihood.eval()

        return self.model, self.likelihood


def gaussian_predict(
    model: gpytorch.models, likelihood: gpytorch.likelihoods, feat: str
) -> tuple[np.ndarray, np.ndarray]:  # FIXME check the type
    """Predict the mean and variance of the GP model for the given feature.
    Args:
        model: The GP model.
        likelihood: The likelihood.
        feat: The feature to predict on.
    Returns:
        pred_mean: The predicted mean.
        pred_var: The predicted variance.
    """
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(feat))
        pred_mean = observed_pred.mean.numpy()
        pred_var = observed_pred.variance.numpy()

    return pred_mean, pred_var


### ======== GAUSSIAN MODELS =================


class CustomGPModel(gpytorch.models.ExactGP):
    """GP model with a constant mean and a RBF x linear kernel."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods,
        md: DatasetHandler,
    ):
        """
        Args:
            train_x: The training features.
            train_y: The training targets.
            likelihood: The likelihood to use.
            md: The DatasetHandler for the training dataset.
        """
        super().__init__(train_x, train_y, likelihood)
        self.md = md
        self.num_outputs = train_y.shape[1]
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=self.num_outputs
        )
        ker_nonlin = gpytorch.kernels.RBFKernel(active_dims=md.dims(md.features))
        kern_lin = gpytorch.kernels.LinearKernel()
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            kern_lin * ker_nonlin, num_tasks=self.num_outputs
        )

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        """Forward pass of the model.
        Args:
            x: The input features.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class GPModelElementwiseKernel(gpytorch.models.ExactGP):
    """GP model using a constant mean and elementwise sequence comparison"""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods,
        md: DatasetHandler,
    ):
        """
        Args:
            train_x: The training features.
            train_y: The training targets.
            likelihood: The likelihood to use.
            md: The DatasetHandler for the training dataset.
        """
        super().__init__(train_x, train_y, likelihood)
        self.md = md
        self.num_outputs = train_y.shape[1]
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=self.num_outputs
        )

        covar_module = OneHotIndexKernel(num_categories=21)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            covar_module, num_tasks=self.num_outputs
        )

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        """Forward pass of the model.
        Args:
            x: The input features.
        """
        mean_x = self.mean_module(x).float()
        covar_x = self.covar_module(x).float()
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class GPModelElementwiseNPositionKernel(gpytorch.models.ExactGP):
    """GP model ElementwiseNPositionKernel"""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods,
        md: DatasetHandler,
    ):
        """
        Args:
            train_x: The training features.
            train_y: The training targets.
            likelihood: The likelihood to use.
            md: The DatasetHandler for the training dataset.
        """
        super().__init__(train_x, train_y, likelihood)
        self.md = md
        self.num_outputs = train_y.shape[1]
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=self.num_outputs
        )
        covar_module = OneHotIndexRBFKernel(num_categories=21)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            covar_module, num_tasks=self.num_outputs
        )

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        """Forward pass of the model.
        Args:
            x: The input features.
        """
        mean_x = self.mean_module(x).float()
        covar_x = self.covar_module(x).float()
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
