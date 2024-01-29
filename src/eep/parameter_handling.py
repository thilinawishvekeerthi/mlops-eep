import datasets as ds
import jax.numpy as np
import numpy as onp
from jax.random import PRNGKey
from pathlib import Path
import pedata.config.paths as paths


def get_test_train(
    df: ds.Dataset,
    rand_seed: PRNGKey,
    max_data: int,
    splt_type: str = "random",
    train_frac: float = 0.8,
):
    """Does different test train splits of the dataset.

    Args:
        df (ds.Dataset): Name fo the dataset to load.
        rand_seed (PRNGKey): Which random seed to use.
        max_data (int): Maximum number of datapoint which should be used.
        splt_type (str): Which method is used for the spiting.
        train_frac (float): Fraction of data that is to be used as training data, whereas the `1 - train_frac` remaining data points will be used for testing. Defaults to 0.8.

    Returns:
        train (pd.DataFrame): Train dataset.
        test (pd.DataFrame): Test dataset.
    """
    # read data

    assert train_frac >= 0.0 and train_frac <= 1.0

    max_data = min(max_data, len(df))
    test_size = int(max_data * (1.0 - train_frac))

    if splt_type == "random":
        splitted = df.train_test_split(
            test_size=test_size,
            train_size=max_data - test_size,
            seed=list(onp.array(rand_seed)),
        )
    elif splt_type == "extrapolation":
        assert False, "this needs to be fixed"  # FIXME
        asc = np.argsort(df.target.values.squeeze())
        train_len = int(len(asc) * train_frac)
        train, test = df[asc[:train_len]], df[asc[train_len:]]

    return splitted


def get_ckpt_path(file_name: str, method: str, param):
    """Set the path for saving the checkpoints of the model.

    Args:
        file_name (str): Name of the file on which the model is trained.
        method (str): Which method is used for the training.
        param (dict): Dict which the parameter of the model.

    Returns:
        str: Filepath of the checkpoints
    """
    try:
        ckpt_dir = Path(param["ckpt_dir"])
    except BaseException:
        ckpt_dir = paths.PE_CHECKPOINTS_DIR
    return ckpt_dir / file_name / method
