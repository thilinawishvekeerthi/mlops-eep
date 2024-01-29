from typing import Union
import pandas as pd
import numpy as np

from datasets import load_dataset
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pedata as ped
import datasets as ds

import jax.random as jr
from eep.models import SeqPropertyPred
import datasets as ds

import eep


def print_results(mean_mse_score, mean_spearman_score, dataset_name=None):
    if dataset_name is not None:
        print("#", dataset_name)
    for k in mean_mse_score:
        print(f"{k}: {mean_mse_score[k]:.3f}, {mean_spearman_score[k]:.3f}")


def get_cv_split(dataset: Union[str, ds.Dataset], k: int = 5):
    if isinstance(dataset, str):
        assert 100 % k == 0, "k must be a factor of 100"
        step_size = 100 // k
        val_ds = ds.load_dataset(
            dataset,
            split=[f"train[{j}%:{j+step_size}%]" for j in range(0, 100, step_size)],
        )
        train_ds = ds.load_dataset(
            dataset,
            split=[
                f"train[:{j}%]+train[{j+step_size}%:]" for j in range(0, 100, step_size)
            ],
        )
        return train_ds, val_ds
    else:
        train_ds = []
        val_ds = []
        for i in range(k):
            spl = dataset.train_test_split(1.0 / k, shuffle=True, seed=i)
            train_ds.append(spl["train"])
            val_ds.append(spl["test"])
        return train_ds, val_ds


def run_cv(model: SeqPropertyPred, train_ds, val_ds, print_pastable: bool = False):
    k = len(train_ds)
    mse_score = {}
    spearman_score = {}
    observed_count = {}
    for key in ped.util.get_target_columns(val_ds[0]):
        mse_score[key] = np.zeros((k, 1))
        spearman_score[key] = np.zeros((k, 1))
        observed_count[key] = np.zeros((k, 1))

    # k fold cross validation
    for j in range(0, k):
        print(j)

        # fit
        model.fit(jr.PRNGKey(j), train_ds[j])
        # predict
        y_pred = model.predict(val_ds[j])

        keys, y_test_all = ped.util.get_target(val_ds[j])
        for i, key in enumerate(keys):
            y_test = y_test_all[:, i]
            not_nan = np.where(
                ~np.isnan(val_ds[j].with_format("numpy")[key].astype(float))
            )[0]
            observed_count[key][j] = len(not_nan)
            y_pred_single_target = y_pred[key]["mean"] + 0.5 * y_pred[key]["var"]
            mse_score[key][j] = mean_squared_error(
                y_test[not_nan], y_pred_single_target[not_nan]
            )
            spearman_score[key][j] = stats.spearmanr(
                y_test[not_nan], y_pred_single_target[not_nan]
            )[0]

    # weighted average per target, weighted by number of samples
    mean_mse_score = {
        key: np.sum([mse_score[key][j] * observed_count[key][j] for j in range(k)])
        / np.sum(observed_count[key])
        for key in mse_score
    }
    mean_spearman_score = {
        key: np.sum([spearman_score[key][j] * observed_count[key][j] for j in range(k)])
        / np.sum(observed_count[key])
        for key in spearman_score
    }
    sum_observed_count = {key: np.sum(observed_count[key]) for key in observed_count}
    mean_mse_score["mean"] = np.sum(
        [mean_mse_score[key] * sum_observed_count[key] for key in mean_mse_score]
    ) / np.sum(list(sum_observed_count.values()))
    mean_spearman_score["mean"] = np.sum(
        [
            mean_spearman_score[key] * sum_observed_count[key]
            for key in mean_spearman_score
        ]
    ) / np.sum(list(sum_observed_count.values()))
    if print_pastable:
        print_results(mean_mse_score, mean_spearman_score)
    return mean_mse_score, mean_spearman_score
