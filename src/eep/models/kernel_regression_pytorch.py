import numpy as np
import datasets as ds
import torch
from torch.utils.data import DataLoader
from collections import namedtuple
from ..util.typing import PRNGKeyT
from jax.random import PRNGKey
from eep.models import SeqPropertyPred
from pedata.util import DatasetHandler, zscore, de_zscore_predictions
from typing import Iterator, Union
import itertools


def reshape_1hot_numpy(
    flat_one_hot_array: np.array, len_seq: int = None, len_aa: int = 21
):
    """
    Reshape the 1-hot encoded amino acid sequence [k_seq x (len_seq, len_aa)]
        into a [k_seq, len_seq, len_aa] numpy array

    Args:
        ds (np.array): flatten 1-hot encoded amino acid sequence
        len_seq (int): sequence length
        len_aa (int): number of amino acids

    Returns:
        emb_list_1hot (np.array): 1-hot encoded amino acid sequence"""

    if len_seq is None:
        len_seq = int(flat_one_hot_array.shape[1] / len_aa)

    return flat_one_hot_array.reshape(flat_one_hot_array.shape[0], len_seq, len_aa)


def kmers_mean_embeddings(emb_list, k, as_torch_tensor=True):
    """Compute the mean embedding of k-mers

    Args:
        emb_list (list): list of embeddings
        k (int): k-mer size
        as_torch_tensor (bool): If True, return a torch tensor. Defaults to True.

    Returns:
        U (torch.Tensor): mean embedding of k-mers
    """
    d = emb_list[0].shape[1]
    U = np.zeros((len(emb_list), k * d))
    for i in range(0, len(emb_list)):
        # print(i)
        n = emb_list[i].shape[0]
        X = np.zeros((n - k + 1, k * d))
        for j in range(0, n - k + 1):
            X[j, :] = emb_list[i][j : j + k, :].reshape(1, -1)
        U[i, :] = np.mean(X, axis=0)

    if as_torch_tensor:
        return torch.from_numpy(U)
    else:
        return U


def l2_norm_kmers_mean_embeddings(emb_list, k, as_torch_tensor=True):
    """Compute the l2 norm of the mean embedding of k-mers

    Args:
        emb_list (list): list of embeddings
        k (int): k-mer size
        as_torch_tensor (bool): If True, return a torch tensor. Defaults to True.

    Returns:
        U_norm (torch.Tensor): l2 norm of the mean embedding of k-mers
    """

    d = emb_list[0].shape[1]
    U = np.zeros((len(emb_list), k * d))
    for i in range(0, len(emb_list)):
        # print(i)
        n = emb_list[i].shape[0]
        X = np.zeros((n - k + 1, k * d))
        for j in range(0, n - k + 1):
            X[j, :] = emb_list[i][j : j + k, :].reshape(1, -1)
        U[i, :] = np.mean(X, axis=0)
    U_norm = U / np.linalg.norm(U, axis=1)[:, None]

    if as_torch_tensor:
        return torch.from_numpy(U_norm)
    else:
        return U_norm


def kmer_embeddings_centers(emb_list_train, k, as_torch_tensor=True):
    """Compute the k-mer embeddings centers

    Args:
        emb_list_train (list): list of embeddings
        k (int): k-mer size
        as_torch_tensor (bool): If True, return a torch tensor. Defaults to True.

    Returns:
            U1 (torch.Tensor): k-mer embeddings centers"""

    seq_length = emb_list_train[0].shape[0]
    d = emb_list_train[0].shape[1]
    n_train = emb_list_train.shape[0]

    kmer_list = []
    for i in range(0, n_train):
        X = np.zeros((seq_length - k + 1, k * d))
        for j in range(0, seq_length - k + 1):
            X[j, :] = emb_list_train[i][j : j + k, :].reshape(1, -1)
        kmer_list.append(X)

    U = np.unique(np.array(list(itertools.chain.from_iterable(kmer_list))), axis=0)
    # l2 norm
    U1 = U / np.linalg.norm(U, axis=1)[:, None]
    if as_torch_tensor:
        return torch.from_numpy(U1)
    else:
        return U1


def linear_kernel_matrix(X, Y):
    """
    Compute the Linear Kernel Matrix between two sets of data points.

    Args:
        X (Tensor): First set of data points (m1, d).
        Y (Tensor): Second set of data points (m2, d).

    Returns:
        kernel_matrix (Tensor): Linear Kernel Matrix (m1, m2).
    """
    return X @ Y.t()


def eigendecomposition(kmat: torch.Tensor, preconditioning_level: int) -> torch.Tensor:
    """
    Eigendecomposition of a kernel matrix

    Args:
        kmat (torch.Tensor): kernel matrix
        preconditioning_level (int): number of eigenvalues to keep

    Returns:
        E (torch.Tensor): eigenvector matrix associated to the q largest eigenvalues
        D (torch.Tensor): eigenvalue matrix associated to the q largest eigenvalues
        d (torch.Tensor): q largest eigenvalues
        lambda_q (torch.Tensor): q-th eigenvalue
    """
    q = preconditioning_level
    eigvals, eigvecs = torch.linalg.eigh(kmat)
    pos = torch.argsort(
        eigvals, descending=True
    )  # sort eigenvalues in descending order
    E = eigvecs[:, pos[0:q]]
    d = eigvals[pos[0:q]]
    D = torch.diag(eigvals[pos[0:q]])
    lambda_q = eigvals[pos[q]]
    # TODO use a namedtuple instead of a tuple?
    # EigenDecomp = namedtuple("EigenDecomp", ["E", "D", "d", "lambda_q"])
    # return EigenDecomp(E, D, d, lambda_q)
    return E, D, d, lambda_q


class EigenPro2:
    """
    Solve the linear system K*alpha = y
    Adapted for a large number of support vectors (m large)
    """

    def __init__(
        self,
        kernel_fn: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        nystrom_size: int,
        preconditioning_level: int,
    ):
        """
        Initialization of the EigenPro algorithm

        Args:
            kernel_fn (torch.Tensor): kernel function
            X (torch.Tensor): training data
            y (torch.Tensor): target values
            nystrom_size (int): subsample size
            preconditioning_level (int): number of eigenvalues to keep
        """
        self.X = X
        self.y = y
        self.nystrom_size = nystrom_size
        self.preconditioning_level = preconditioning_level
        self.kernel_fn = kernel_fn

    def setup(self) -> torch.Tensor:
        """
        Setup method for the EigenPro algorithm
        Made up of the following steps:
        - subsampling
        - eigendecomposition
        - preconditioning matrix computation
        - batch size computation
        - learning rate computation

        Returns:
            idx_subsample (torch.Tensor): subsample indices
            Xsub (torch.Tensor): subsample data
            E (torch.Tensor): eigenvectors
            Dx (_type_): preconditioning matrix
            eta (float): learning rate
            m (int): batch size
        """
        device = self.X.device
        idx_subsample = np.random.choice(
            np.arange(0, self.X.shape[0]), self.nystrom_size, replace=False
        )  # subsample indices
        idx_subsample = torch.from_numpy(idx_subsample)  # torch conversion
        Xsub = self.X[idx_subsample, :]  # subsample data
        Ksub = self.kernel_fn(Xsub, Xsub)  # kernel matrix of subsample data
        E, _, d, lambda_q = eigendecomposition(Ksub, self.preconditioning_level)
        Dx = (
            (1 / self.nystrom_size)
            * torch.diag(1 / d).to(device)
            @ (
                torch.eye(self.preconditioning_level, device=device)
                - lambda_q * torch.diag(1 / d)
            ).to(device)
        )

        # batch size
        m = int(np.ceil(min(1 / lambda_q.item(), self.nystrom_size)))
        # learning rate
        if m <= 1 / lambda_q.item():
            eta = 1 / (2 * m)
        else:
            eta = (0.99 * m) / (1 + (m - 1) * lambda_q.item())

        # TODO use a namedtuple instead of a tuple?
        # EigenPro2 = namedtuple("EigenPro2", ["idx_subsample", "Xsub", "E", "eta", "m"])
        # return EigenPro2(idx_subsample, Xsub, E, Dx, eta, m)
        return idx_subsample, Xsub, E, Dx, eta, m

    def iteration(
        self,
        Xsub: torch.Tensor,
        E: torch.Tensor,
        Dx: torch.Tensor,
        eta: float,
        m: int,
        alpha: float,
        idx_subsample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Iteration method for the EigenPro algorithm

        Args:
            Xsub (torch.Tensor): subsample data from the setup method
            E (torch.Tensor): eigenvectors from the setup method
            Dx (torch.Tensor): preconditioning matrix from the setup method
            eta (torch.Tensor): learning rate from the setup method
            m (torch.Tensor): batch size from the setup method
            alpha (torch.Tensor): regression coefficients
            idx_subsample (torch.Tensor): subsample indices from the setup method

        Returns:
            alpha (torch.Tensor): regression coefficients
        """
        idx_batch = np.random.choice(np.arange(self.X.shape[0]), m, replace=False)
        idx_batch = torch.from_numpy(idx_batch)
        Xbatch = self.X[idx_batch, :]
        ybatch = self.y[idx_batch, :]
        Kxz = self.kernel_fn(Xbatch, self.X)
        Ksub_batch = self.kernel_fn(Xsub, Xbatch)

        g = Kxz @ alpha - ybatch  # stochastic gradient
        alpha[idx_batch] = alpha[idx_batch] - eta * g  # gradient step
        alpha[idx_subsample] = (
            alpha[idx_subsample] + eta * E @ Dx @ E.t() @ Ksub_batch @ g
        )  # gradient correction

        return alpha

    def fit(self, nb_iter: int) -> torch.Tensor:
        """
        Fit method for the EigenPro algorithm

        Args:
            nb_iters (int): number of iterations of the gradient descent

        Returns:
            alpha (torch.Tensor): regression coefficients
        """
        idx_subsample, Xsub, E, Dx, eta, m = self.setup()
        alpha = torch.zeros(
            self.X.shape[0],
            1,
            requires_grad=False,
            dtype=torch.float64,
            device=self.X.device,
        )

        for _ in range(nb_iter):
            alpha = self.iteration(Xsub, E, Dx, eta, m, alpha, idx_subsample)
            return alpha


class KernelRegression(SeqPropertyPred):
    """
    Kernel Regression class
    """

    def __init__(
        self,
        feature_list: list[str] = None,
        kernel_fn: torch.Tensor = None,
        centers: torch.Tensor = None,
        scoring_fn: torch.Tensor = None,
        device=None,
        verbose: bool = True,
    ):
        """
        Initialization of the KernelRegression class
            Args:
                feature_list (list[str]): List of features to use.
                    Defaults to None.
                kernel_fn (torch.Tensor): kernel function
                centers (torch.Tensor): support vectors
                scoring_fn (torch.Tensor): target scoring function
                preconditioning_level_training (int): number of eigenvalues to keep from the eigendecomposition of kernel_fn(X,X) of training data.
                epochs: Number of epochs to train the kernel.
                    Defaults to 1000.
                learning_rate_training (float): learning rate for the gradient descent of the training data
                nystrom_size_training (int): size of the subsample of the training data
                batch_size_training: batch size for the gradient descent of the training data
                nystrom_size_centers: subsample size for the EigenPro algorithm associated to the support vectors
                preconditioning_level_centers: number of eigenvalues to keep for the EigenPro algorithm associated to the support vectors.
                target_standardization (bool): If True, will standardize the target values (substract the mean and divide by standard deviation).
                    Default to False.
            Raises:
                ValueError: If feature_list is None.

            Notes:
                `preconditioning_level_training` depends mostly about
                    - nystrom_size_training
                    - or the training data size such that preconditioning_level_training <= min(nystrom_size_training, X.shape[0])
                `preconditioning_level_centers` depends mostly about
                    - of nystrom_size_centers such that preconditioning_level_centers <= nystrom_size_centers

            Examples:
                >>> from datasets import load_dataset
                >>> from eep.models import KernelRegression
                >>> from eep.plot import plot_pred
                >>> import numpy as np
                >>> from pedata.util import get_target
                >>> dataset_name = "Company/CrHydA1_PE_REGR"
                >>> dataset = load_dataset(dataset_name)["train"]
                >>> dataset = dataset.train_test_split(0.2, seed=42)

                >>> # kernel regression example
                >>> KR = KernelRegression(feature_list=["aa_1hot"], target_standardization=True, epochs=10, verbose=False)
                >>> print(KR) # doctest: +NORMALIZE_WHITESPACE
                KernelRegression(feature_list=['aa_1hot'], centers=None, scoring_fn=MSELoss(), nystrom_size_training=84, batch_size_training=84, nystrom_size_centers=10, preconditioning_level_centers=5, target_standardization=True, epochs=10, learning_rate_training=100.0, nb_iter_centers=100, preconditioning_level_training=10, verbose=False)
                >>> KR.fit(rng=0, train_ds=dataset["train"])
                >>> y_pred = KR.predict(test_ds=dataset["test"])


        """
        super().__init__()

        if feature_list is None:
            raise ValueError("feature_list must be specified")

        if kernel_fn is None:
            kernel_fn = linear_kernel_matrix

        if scoring_fn is None:
            scoring_fn = torch.nn.MSELoss()

        self.feat_list = feature_list
        self.kernel_fn = kernel_fn
        self.centers = centers
        self.scoring_fn = scoring_fn
        self.verbose = verbose

        self.size_kmer = 1

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if verbose:
            if torch.cuda.is_available():
                print(f"Current cuda device: {torch.cuda.get_device_name(0)}")
            else:
                print(f"No GPU available, using {device} instead.")

        self.device = device

    def __repr__(self):
        if hasattr(self, "batch_size_training"):
            return (
                f"KernelRegression("
                f"feature_list={self.feat_list}, "
                f"centers={self.centers}, "
                f"scoring_fn={self.scoring_fn}, "
                f"nystrom_size_training={self.nystrom_size_training}, "
                f"batch_size_training={self.batch_size_training}, "
                f"nystrom_size_centers={self.nystrom_size_centers}, "
                f"preconditioning_level_centers={self.preconditioning_level_centers}, "
                f"target_standardization={self.target_standardization}, "
                f"epochs={self.epochs}, "
                f"learning_rate_training={self.learning_rate_training}, "
                f"nb_iter_centers={self.nb_iter_centers}, "
                f"preconditioning_level_training={self.preconditioning_level_training}, "
                f"verbose={self.verbose}"
                ")"
            )
        else:
            return (
                f"KernelRegression("
                f"feature_list={self.feat_list}, "
                f"centers={self.centers}, "
                f"scoring_fn={self.scoring_fn}, "
                f"verbose={self.verbose}"
                ")"
            )

    def to(self):
        """Change the device of the CustomKernelModel class

        Args:
            device (torch.device): device to which the CustomKernelModel class is sent (cpu or gpu)

        Returns:
            self (CustomKernelModel): CustomKernelModel class sent to the device
        """
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.centers = self.centers.to(self.device)

    @property
    def features(self) -> Iterator[str]:
        """Return the names of the features used by the model.

        Yields:
            Iterator[str]: The feature names.
        """
        yield from self.feat_list

    def get_features(
        self, d: ds.Dataset
    ) -> torch.Tensor:  # FIXME : check if this is necessary and/or the right way
        """Get the features from the dataset.

        Args:
            d (ds.Dataset): Get the features defined in self.feat_list from Dataset d.

        Returns:
            torch.Tensor: The features.
        """
        # get all features in self.feat_list from the dataset, reshape to (n_samples, -1)
        # then concatenate them to a single array
        md_features = DatasetHandler(d, self.feat_list)
        return md_features.cat(d).to(torch.float32), md_features

    def fit(
        self,
        train_ds: ds.Dataset,
        epochs: int = 10000,
        learning_rate_training: float = 100.0,
        nystrom_size_training: int = None,
        batch_size_training: int = None,
        nb_iter_centers: int = 100,
        nystrom_size_centers: int = None,  # 10,
        preconditioning_level_centers: int = None,  # 5,
        preconditioning_level_training: int = None,  # 10
        target_standardization: str = False,
        rng: PRNGKeyT = PRNGKey(0),
    ) -> None:
        """Train the model on the given dataset.

        Args:
            train_ds: Training dataset.
            epochs: Number of epochs to train the kernel.
                Defaults to 1000.
            learning_rate_training: Learning rate for the gradient descent of the training data
            nystrom_size_training: Size of the subsample of the training data
            batch_size_training: Batch size for the gradient descent of the training data
            nb_iter_centers: Number of iterations of the gradient descent for the support vectors
            nystrom_size_centers: Subsample size for the EigenPro algorithm associated to the support vectors
            preconditioning_level_centers: Number of eigenvalues to keep for the EigenPro algorithm associated to the support vectors.
            preconditioning_level_training: Number of eigenvalues to keep from the eigendecomposition of kernel_fn(X,X) of training data.
            rng: Random number generator seed. Defaults to PRNGKey(0).
        """

        if nystrom_size_training is None:
            nystrom_size_training = len(train_ds)
        self.nystrom_size_training = nystrom_size_training

        if batch_size_training is None:
            batch_size_training = len(train_ds)
        self.batch_size_training = batch_size_training

        if nystrom_size_centers is None:
            nystrom_size_centers = 6
        self.nystrom_size_centers = nystrom_size_centers

        if preconditioning_level_centers is None:
            preconditioning_level_centers = 3
        self.preconditioning_level_centers = preconditioning_level_centers

        if preconditioning_level_training is None:
            preconditioning_level_training = 6
        self.preconditioning_level_training = preconditioning_level_training

        self.target_standardization = target_standardization
        self.epochs = epochs
        self.learning_rate_training = learning_rate_training
        self.nb_iter_centers = nb_iter_centers

        self._store_target_names(train_ds)
        _, y_train = self.get_target(train_ds, as_dict=False)

        if self.target_standardization:
            self.y_train_mean = np.nanmean(y_train, axis=0, keepdims=True)
            self.y_train_std = np.nanstd(y_train, axis=0, keepdims=True)

            y_train = zscore(y_train)

        # self.train_target = y_train
        # self.best_train_pred_mean = y_train.max()

        X_train, _ = self.get_features(train_ds)

        self.centers = kmer_embeddings_centers(
            reshape_1hot_numpy(X_train), self.size_kmer, as_torch_tensor=True
        )

        X_train_l2 = l2_norm_kmers_mean_embeddings(
            reshape_1hot_numpy(X_train), self.size_kmer, as_torch_tensor=True
        )

        self._fit_batch_with_inexact_projection(
            X_train=X_train_l2,
            y_train=y_train,
        )

    def _fit_batch_with_inexact_projection(
        self,
        X_train: torch.Tensor,
        y_train: [torch.Tensor | np.ndarray],
    ) -> torch.Tensor:
        """

        Batch version of the fit method for general kernel regression model
        Adapted for large datasets and large number of support vectors

        Args:
            X_train (torch.Tensor): training data
            y_train (torch.Tensor): target values

        """

        if type(y_train) == np.ndarray:
            y_train = torch.from_numpy(y_train).to(torch.float64)

        if y_train.dtype == torch.float32:
            y_train = y_train.to(torch.float64)

        if self.preconditioning_level_training is None:  # FIXME temporary
            self.preconditioning_level_training = min(X_train.shape[0] - 1, 1000)

        self.X_train = X_train
        self.y_train = y_train
        device = X_train.device

        idx_sub_training = torch.from_numpy(
            np.random.choice(
                self.X_train.shape[0], self.nystrom_size_training, replace=False
            )
        )  # subsample indices # torch conversion

        Xsub_training = self.X_train[idx_sub_training, :]  # subsample data

        Ksub_sub_training = self.kernel_fn(
            Xsub_training, Xsub_training
        )  # kernel matrix of subsample data

        E, _, d, lambda_q = eigendecomposition(
            Ksub_sub_training, self.preconditioning_level_training
        )  # eigendecomposition

        K_centers_sub_training = self.kernel_fn(
            self.centers, Xsub_training
        )  # kernel matrix between support vectors and subsample data

        C = (
            K_centers_sub_training
            @ E
            @ (
                torch.diag(1 / d).to(device)
                - lambda_q * torch.diag(1 / (d**2)).to(device)
            )
            @ E.t()
        )  # preconditioning matrix

        self.beta = torch.zeros(
            self.centers.shape[0],
            1,
            requires_grad=False,
            dtype=torch.float64,
            device=device,
        )

        loader = DataLoader(
            list(zip(self.X_train, self.y_train)),
            shuffle=True,
            batch_size=self.batch_size_training,
        )  # data loader

        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                Kbatch_training_centers = self.kernel_fn(X_batch, self.centers)
                Ksub_training_batch_training = self.kernel_fn(Xsub_training, X_batch)

                g = Kbatch_training_centers @ self.beta - y_batch
                h = (
                    Kbatch_training_centers.t() @ g
                    - C @ Ksub_training_batch_training @ g
                )
                self.theta = EigenPro2(
                    self.kernel_fn,
                    self.centers,
                    h,
                    self.nystrom_size_centers,
                    self.preconditioning_level_centers,
                ).fit(self.nb_iter_centers)
                self.beta -= self.learning_rate_training * self.theta

            y_pred_train = self.forward(self.X_train, self.beta)
            scr_train = self.scoring_fn(y_pred_train, self.y_train)
            if self.verbose:
                print("epoch: ", epoch, "train_loss: ", scr_train.item())

    # FIXME the .forward() may be used better ??
    def forward(self, X: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Forward pass of the kernel regression model

        Args:
            X (torch.Tensor): test data
            beta (torch.Tensor): regression coefficients

        Returns:
            y_pred (torch.Tensor): predicted values
        """

        Kxz = self.kernel_fn(X, self.centers)
        y_pred = Kxz @ beta
        return y_pred

    def predict(
        self,
        test_ds: Union[ds.Dataset, dict[str, np.ndarray]],
        flat: bool = False,
        tolist: bool = False,
    ) -> Union[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
        """Prediction method for the kernel regression model

        Args:
            test_ds (Union[ds.Dataset, dict[str, np.ndarray]]): test data
            flat (bool): If True, return a flat dictionary.
                Defaults to False.
            tolist (bool): If True, return a dictionary with lists instead of numpy arrays.

        Returns:
            y_pred (torch.Tensor): predicted values
        """
        X_test, _ = self.get_features(test_ds)

        X_test_l2 = l2_norm_kmers_mean_embeddings(
            reshape_1hot_numpy(X_test), self.size_kmer, as_torch_tensor=True
        )
        X_test_l2 = X_test_l2.to(self.X_train.device)

        y_pred_test = self.forward(X_test_l2, self.beta)

        if self.target_standardization:
            # de-zscore

            y_pred_test = de_zscore_predictions(
                zs_pred_means=y_pred_test.cpu().numpy(),
                mean=self.y_train_mean,
                std=self.y_train_std,
            )  # FIXME: check if this is correct

        return self._prediction_arrays_to_dictionary(
            y_pred_test, flat=flat, tolist=tolist
        )


if __name__ == "__main__":
    pass
