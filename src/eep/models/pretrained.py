"""
This module contains the pretrained models for enzyme efficiency prediction.
Both classification and regression models are available.
These models have deep transformer architectures.
You will need GPUs to run these models.
Make sure you have CUDA installed and PyTorch with CUDA support is installed.

Available models:
    - LowerGrowthTemperaturePred: Predicts the lower growth temperature of a given sequence. It is trained on the dataset from https://huggingface.co/datasets/Company/TemStaPro-Minor-bal65
    - SolubilityPred: Predicts the solubility of a given sequence. It is trained on the dataset from https://huggingface.co/datasets/Company/Proteinea_solubility_v2

Example:
    >>> from eep.models import LowerGrowthTemperaturePred, SolubilityPred
    >>> from datasets import load_dataset
    >>> dataset_name = "Company/TemStaPro-Minor-bal65"
    >>> dataset = load_dataset(dataset_name, split='testing[:50]')
    >>> model = LowerGrowthTemperaturePred(['ankh'])
    >>> predictions = model.predict(test_df=dataset)
    >>> print(predictions['growth_temp']['mean'].shape)
    (50,)

    >>> dataset_name = "Company/Proteinea_solubility_v2"
    >>> dataset = load_dataset(dataset_name, split='test[:50]')
    >>> model = SolubilityPred(['ankh'])
    >>> predictions = model.predict(test_df=dataset)
    >>> print(predictions['solubility'].shape)
    (50,)
"""
import os
from tqdm import tqdm
from typing import Iterator
import datasets as ds
import torch
import numpy as np
from pedata.pytorch_dataloaders import Dataloader
from .base import SeqPropertyPred, SeqClassifier, PreTrained
from .utils import load_model
from .pytorch_models import Transformer, TransformerS

# from jaxrk.core.typing import PRNGKeyT
# from .pytorch_models import Transformer, TransformerS
# import datasets as ds
# from .base import SeqPropertyPred, SeqClassifier, PreTrained
# from .utils import load_model
# import torch
# from typing import Union, Iterator
# import numpy as np
# from tqdm import tqdm
# from pedata.pytorch_dataloaders import Dataloader
# import os


class LowerGrowthTemperaturePred(SeqPropertyPred, PreTrained):
    """
    Predicts the lower growth temperature of a given sequence.
    """

    def __init__(
        self,
    ) -> None:
        self.load_weights()

    def load_weights(self) -> None:
        """
        Downloads weights from gcloud storage and loads them into the model.
        """
        self.model = load_model("transformer_4_temstapro_lowergrowthtemperature")
        self.model.eval()

    @property
    def trained_dataset(self) -> str:
        """
        Return the name of the dataset the model is trained on.

        Returns:
            str: The name of the dataset.
        """
        return "Company/TemStaPro-Minor-bal65"
    
    @property
    def trained_split(self) -> str:
        """
        Return the name of the split the model is trained on.

        Returns:
            str: The name of the split.
        """
        return "source_split"

    @property
    def features(self) -> Iterator[str]:
        """
        Return the names of the features used by the model.

        Yields:
            Iterator: The feature names.
        """
        yield from ["ankh"]

    @property
    def target_name(self) -> Iterator[str]:
        """
        Return the names of the targets the model predicts.

        Yields:
            Iterator[str]: The feature names.
        """
        yield from ["growth_temp"]

    def fit(self) -> Transformer:
        """
        Dummy method. This model is not trainable. It is pre-trained.

        Returns:
            Transformer: The model used for prediction.
        """
        return self.model

    def predict(
        self,
        test_df: ds.Dataset | dict[str, np.ndarray],
        flat: bool = False,
        tolist: bool = False,
    ) -> dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]]:
        """
        Predicts the mean and variance of the lower growth temperature of the given sequences.

        Note: This model is trained on the dataset from https://huggingface.co/datasets/Company/TemStaPro-Minor-bal65

        Args:
            test_df: Dataset for which to predict target variables.
            flat: Whether to return the dictionary with the arrays as values or as a dictionary of dictionaries. Defaults to False.
            tolist: Whether to convert the arrays to lists. Defaults to False.

        Returns:
            A dictionary, where keys are the different target label columns from the dataset. Values are the predicted mean and variance of the target label.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_loader = Dataloader(
            dataset=test_df,
            embedding_names=["ankh"],
            batch_size=1,
            device=device,
            shuffle=False,
        )
        mean = []
        vars = []
        with torch.no_grad():
            for features in tqdm(data_loader):
                X = features["ankh"]
                y_pred = self.model(X)
                mean.append(y_pred.cpu().numpy())
                vars.append(torch.zeros_like(y_pred).cpu().numpy())
        mean = np.concatenate(mean).squeeze()
        vars = np.concatenate(vars).squeeze()
        return self._prediction_arrays_to_dictionary(
            mean, vars, flat=flat, tolist=tolist
        )


class SolubilityPred(SeqClassifier, PreTrained):
    """
    Predicts the solubility of a given sequence.
    The output is a logit. Higher the number is more soluble the protein is.

    Notes: The model is trained on the dataset from https://huggingface.co/datasets/Company/Proteinea_solubility_v2
    """

    def __init__(
        self,
    ) -> None:
        self.load_weights()

    def load_weights(self) -> None:
        """
        Downloads weights from gcloud storage and loads them into the model.
        """
        self.model = load_model("transformerS_4_DeepSolV2_solubility")
        self.model.eval()

    @property
    def trained_dataset(self) -> str:
        """
        Return the name of the dataset the model is trained on.

        Returns:
            str: The name of the dataset.
        """
        return "Company/Proteinea_solubility_v2_PE_CLASS"

    @property
    def trained_split(self) -> str:
        """
        Return the name of the split the model is trained on.

        Returns:
            str: The name of the split.
        """
        return "source_split"

    @property
    def features(self) -> Iterator[str]:
        """
        Return the names of the features used by the model.

        Yields:
            The feature names.
        """
        yield from ["ankh"]

    @property
    def target_name(self) -> Iterator[str]:
        """
        Return the names of the targets the model predicts.

        Yields:
            The feature names.
        """
        yield from ["solubility"]

    def fit(self) -> TransformerS:
        """
        Dummy method. This model is not trainable. It is pre-trained.
        """
        return self.model

    def _prediction_arrays_to_dictionary(
        self, arrays: list[np.ndarray], tolist: bool = False
    ) -> dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]]:
        """Converts the arrays of predictions to a dictionary.

        Args:
            arrays: List of arrays of predictions.
            flat: Whether to return the dictionary with the arrays as values or as a dictionary of dictionaries. Defaults to False.
            tolist: Whether to convert the arrays to lists. Defaults to False.

        Returns:
            The dictionary of predictions.
        """
        dict_ = {}
        for i, target_name in enumerate(self.target_name):
            dict_[target_name] = arrays[:, i].tolist() if tolist else arrays[:, i]
        return dict_

    def save_weights(cls, path: os.PathLike) -> None:
        """
        Dummy method
        """

        raise NotImplementedError

    def predict(self, test_df: ds.Dataset) -> dict[str, np.ndarray]:
        """Predict the values for the 'test_df' based on the trained model.

        Args:
            test_df: Dataset for which to predict target variables.

        Returns:
            A dictionary, where keys are the different target label columns from the dataset.
            Values are the predicted probability of the target label being 1.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_loader = Dataloader(
            dataset=test_df,
            embedding_names=["ankh"],
            batch_size=1,
            device=device,
            shuffle=False,
        )
        logits = []
        with torch.no_grad():
            for features in tqdm(data_loader):
                X = features["ankh"]
                y_pred = self.model(X)
                logits.append(y_pred.cpu().numpy())
        logits = np.concatenate(logits)
        return self._prediction_arrays_to_dictionary(logits, tolist=False)
