import hashlib
import warnings
import os
import io
from typing import List, Optional, Tuple
from .pytorch_models import Transformer, TransformerS
import torch
from google.cloud import storage
from .config import _MODELS

_ARCHITECTURES = {"transformer": Transformer, "transformerS": TransformerS}


def download_blob(
    bucket_name: str, source_blob_name: str, destination_file_name: str
) -> bool:
    """
    Downloads a blob from the google cloud storage pretrained model bucket.

    Args:
        bucket_name (str): Name of the google cloud bucket.
        source_blob_name (str): Name of the blob file in the google cloud bucket.
        destination_file_name (str): Name of the destination file.

    Returns:
        bool: True if the blob was downloaded successfully.

    Example:
        >>> import os
        >>> home = os.path.expanduser("~")
        >>> os.makedirs(os.path.join(home, '.cache', 'Company'), exist_ok=True)
        >>> destination_file_name = os.path.join(home, '.cache', 'Company', '1679EF37A2B71E268A998D3D4A86EF5D9D15AD3CE4696EA5DE331F33FFF73303_transformer_4_temstapro_lowergrowthtemperature.h5')
        >>> bucket_name = "Company-pre-trained-weights"
        >>> source_blob_name = "1679EF37A2B71E268A998D3D4A86EF5D9D15AD3CE4696EA5DE331F33FFF73303_transformer_4_temstapro_lowergrowthtemperature.h5"
        >>> download_blob(bucket_name, source_blob_name, destination_file_name)
        True
    """

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    return True


def get_model(model_name: str) -> Tuple[type, int]:
    """
    Returns the model class for the given model name.

    Args:
        model_name (str): Name of the model.

    Returns:
        tuple(type, int): The model class for the given model name.

    Raises:
        KeyError: If the model is not available.

    Example:
        >>> model_class = get_model('transformer_4_temstapro_lowergrowthtemperature')
        >>> model_class
        (<class 'eep.models.pytorch_models.Transformer'>, 4)
    """
    model_type, num_layers = model_name.split("_")[:2]
    model_class = _ARCHITECTURES.get(model_type)

    return model_class, int(num_layers)


def _download(
    bucket_name: str, source_blob_name: str, root: str, in_memory: bool
) -> bytes | str:
    """
    Download a file from the google cloud storage pretrained model bucket to a local directory and do a SHA256 checksum.

    Args:
        bucket_name (str): Name of the google cloud bucket.
        source_blob_name (str): Name of the blob file in the google cloud bucket.
        root (str): path to download the model files
        in_memory (bool): whether to preload the model weights into host memory

    Returns:
        Union[bytes, str]: the downloaded file bytes if `in_memory` is True, or the path to the downloaded file if `in_memory` is False

    Raises:
        RuntimeError: If the downloaded file does not match the SHA256 checksum
        RuntimeError: If the downloaded file is not a regular file

    Example:
        >>> root = '.cache/Company'
        >>> in_memory = False
        >>> bucket_name = "Company-pre-trained-weights"
        >>> source_blob_name = "1679EF37A2B71E268A998D3D4A86EF5D9D15AD3CE4696EA5DE331F33FFF73303_transformer_4_temstapro_lowergrowthtemperature.h5"
        >>> model_bytes = _download(bucket_name, source_blob_name, root, in_memory)
        >>> model_bytes.split('/')[-1]
        '1679EF37A2B71E268A998D3D4A86EF5D9D15AD3CE4696EA5DE331F33FFF73303_transformer_4_temstapro_lowergrowthtemperature.h5'
    """
    home = os.path.expanduser("~")
    os.makedirs(os.path.join(home, root), exist_ok=True)

    expected_sha256 = source_blob_name.split("_")[0]

    download_target = os.path.join(home, root, source_blob_name)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest().upper() == expected_sha256:
            return model_bytes if in_memory else download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    download_blob(bucket_name, source_blob_name, download_target)
    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest().upper() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    return model_bytes if in_memory else download_target


def available_models() -> List[str]:
    """
    Returns the names of available models

    Returns:
        List[str]: The names of available models

    Example:
        >>> type(available_models())
        <class 'list'>
    """
    return list(_MODELS.keys())


def load_model(
    name: str,
    device: Optional[str | torch.device] = None,
    download_root: str = None,
    in_memory: bool = False,
) -> Transformer | TransformerS:
    """
    Load a Pretrained Model. If the requested model is available locally, it will be loaded from the local cache. Otherwise, it will be downloaded from the cloud bucket.

    Args:
        name (str): one of the official model names listed by `utils.available_models()`
        device (Union[str, torch.device], optional): the PyTorch device to put the model into. Defaults to None.
        download_root (str, optional): path to download the model files. Defaults to None, which will result in using "~/.cache/Company".
        in_memory (bool, optional): whether to preload the model weights into host memory. Defaults to False.

    Returns:
        Union[Transformer, TransformerS]: The loaded model

    Raises:
        RuntimeError: If the model is not available

    Example:
        >>> model = load_model('transformer_4_temstapro_lowergrowthtemperature')
        >>> model
        Transformer(
          (encoder): TransformerEncoder(
            (layers): ModuleList(
              (0-3): 4 x TransformerEncoderLayer(
                (self_attn): MultiheadAttention(
                  (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
                )
                (linear1): Linear(in_features=768, out_features=3072, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (linear2): Linear(in_features=3072, out_features=768, bias=True)
                (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout1): Dropout(p=0.1, inplace=False)
                (dropout2): Dropout(p=0.1, inplace=False)
              )
            )
          )
          (fc1): Linear(in_features=768, out_features=64, bias=True)
          (fc2): Linear(in_features=64, out_features=1, bias=True)
        )
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        download_root = os.path.join(os.path.expanduser("~"), ".cache", "Company")

    if name in available_models():
        checkpoint_file = _download(
            _MODELS[name]["bucket_name"],
            _MODELS[name]["source_blob_name"],
            download_root,
            in_memory,
        )
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    model_class, num_layers = get_model(name)
    model = model_class(n_encoder_layers=num_layers)
    model.load_state_dict(checkpoint)

    return model.to(device)
