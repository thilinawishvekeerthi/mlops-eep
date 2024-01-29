import io
import os
from pathlib import Path
import tqdm
import datasets as ds
import jax.numpy as np
from jax.random import PRNGKey, split
import numpy as onp
from scipy.stats import spearmanr, kendalltau
from .util import Array, PRNGKeyT
from .policies.base import BatchOptPolicy
from .util.artif_fitness import ArtificialFitness
from .parameter_handling import get_test_train
from .models import SeqPropertyPred
from . import models
from .plot import plot_pred

# FLAGS = flags.FLAGS

# flags.DEFINE_string('path', str(Path.home() / "Downloads"), 'Path under which to save plots.')


def print_metrics(metr: dict[str, float], f: io.TextIOWrapper):
    """Print standard metrics to a file.

    Args:
        metr: Dictionary with metrics.
        f: File to write to.
    """
    for k in metr:
        f.write(k + "\n")
        for m in ["MAE", "RMSE", "ρ", "τ", "top100"]:
            f.write(f"\t{m}:\t {metr[k][m]}\n")


def metrics(
    pred: dict[str, dict[str, Array]],
    gt: dict[str, Array],
    include_raw_info: bool = True,
) -> dict[str, Array]:
    # TODO: potentially use Huggingface dataset metrics.
    """Compute metrics such as MAE, RMSE, Spearmans ρ and Kendalls τ from a prediction dict and a ground truth dict

    Args:
        pred (dict[str, dict[str, Array]]): Dictionary with keys that are targets. Value is a dict containing values for keys "mean" and "std".
        gt (dict[str, Array]): Dictionary with keys that are targets, actual measurements as values.
        include_raw_info (bool): Wether to store the raw prediction and ground truth information per target variable in the return value. Defaults to True.
    Returns:
        Metrics for each target varuable
    """
    rval = {}
    for k in pred.keys():
        assert pred[k]["mean"].size == gt[k].size
        rval[k] = {}
        rval[k]["MAE"] = float(np.abs(pred[k]["mean"].ravel() - gt[k].ravel()).mean())
        rval[k]["RMSE"] = float(
            np.sqrt(np.power(pred[k]["mean"].ravel() - gt[k].ravel(), 2).mean())
        )
        dec_crit = (
            pred[k]["mean"].ravel()
            + 0.5 * np.sqrt(np.clip(pred[k]["var"], 0, None)).ravel()
        )
        rval[k]["ρ"] = spearmanr(dec_crit, gt[k].ravel()).correlation
        rval[k]["τ"] = kendalltau(dec_crit, gt[k].ravel()).correlation
        rval[k]["top100"] = (
            len(
                set(onp.argsort(dec_crit.tolist())[-100:]).intersection(
                    onp.argsort(gt[k].ravel().tolist())[-100:]
                )
            )
            / 100
        )
        if include_raw_info:
            rval[k]["gt"] = gt[k].ravel()
            rval[k]["pred_mean"] = pred[k]["mean"]
            rval[k]["pred_var"] = pred[k]["var"]
    return rval


def estimate_and_plot(
    train_ds: ds.Dataset,
    test_ds: ds.Dataset,
    method: models.SeqPropertyPred,
    save: Path = None,
    plot_style="uncertainty",
    rng=PRNGKey(0),
    optimism: float = 0.5,
) -> dict:
    """Model training, prediction, visualization and writing out results.
    # TODO:take apart training and plotting

    Args:
        train_ds: Training dataset.
        test_ds: Testing dataset.
        method: Model to use for training and predictions.
        save: Results saving Path. Defaults to None, in which case results are not recorded.
        plot_style: Plotting style string. Defaults to "uncertainty".
        rng: Jax random number key to use. Defaults to PRNGKey(0).
        optimism: Factor for incorporating standard deviation into UCB criterion. Defaults to 0.5.

    Returns:
        A dictionary containing estimation results, such as predicted mean and variance, as well as quality criteria such as errors and rank correlations.
    """
    # model training
    method.fit(rng, train_ds)
    rval = {}

    if save is not None and not os.path.exists(save) and os.path.isdir(save):
        print("Creating ", save)
        os.makedirs(save / "")
    else:
        print("Results directory exists")
    if save is None:
        f = io.StringIO()
    else:
        f = open(f"{save}_results.txt", "w+")

    rval = {}
    m = np.arange(len(train_ds))
    gt = method.get_target(train_ds, as_dict=True)
    pred = method.predict(train_ds)
    rval["train"] = metrics(pred, gt)

    if save is not None:
        for k in gt:
            for prio_rescale in (True, False):
                fn = "_rescaled" if prio_rescale else ""
                plot_pred(
                    gt[k].ravel(),
                    m,
                    pred[k]["mean"].ravel(),
                    pred[k]["var"].ravel(),
                    f"{save}_{k}_train{fn}.png",
                    target=k,
                    style=plot_style,
                    show=True,
                    optimism=optimism,
                    prioritization_rescale=prio_rescale,
                )

    print("# Train", file=f)
    print_metrics(rval["train"], f)

    # loaded_method = pickel.load(open('Test.sav'),'r')
    if test_ds is not None:
        m = np.arange(len(test_ds))
        gt = method.get_target(test_ds, as_dict=True)
        pred = method.predict(test_ds)
        rval["test"] = metrics(pred, gt)
        print("# Test", file=f)
        print_metrics(rval["test"], f)
        if save is not None:
            for k in gt:
                for prio_rescale in (True, False):
                    fn = "_rescaled" if prio_rescale else ""
                    plot_pred(
                        gt[k].ravel(),
                        m,
                        pred[k]["mean"].ravel(),
                        pred[k]["var"].ravel(),
                        f"{save}_{k}_test{fn}.png",
                        target=k,
                        style=plot_style,
                        show=True,
                        optimism=optimism,
                        prioritization_rescale=prio_rescale,
                    )

    f.seek(0)
    print(f.read())
    f.close()
    return rval


def check_predictions(
    df: ds.Dataset,
    model: SeqPropertyPred,
    rng: PRNGKeyT = PRNGKey(0),
    train_frac: float = 0.8,
    optimism: float = 0.5,
    resultsfile: str | os.PathLike = None,
    **param,
):
    """Split data into train and test. Then train, predict and plot predictions as well as prediction quality metrics.

    Args:
        df: Huggingface dataset object.
        model: Model to use for predictions.
        rng: A random number generator key. Defaults to PRNGKey(0).
        train_frac : Fraction of data that is to be used as training data, whereas the `1 - train_frac` remaining data points will be used for testing.
            Defaults to 0.8.
        optimism: Optimism parameter for UCB. Defaults to 0.5.
        resultsfile: Results file base name.
    """

    # get rand key
    key1, key2 = split(rng, 2)  # FIXME use eep function to split the key

    # get test and train data

    splitted = get_test_train(
        df,
        key1,
        param["ndata"],
        splt_type="random",
        train_frac=train_frac,
    )
    del param["ndata"]
    # print("Comb", train.variant.str.count("_"), test.variant.str.count("_"))

    return estimate_and_plot(
        splitted["train"],
        splitted["test"],
        model,
        save=resultsfile,
        rng=key2,
        optimism=optimism,
        **param,
    )


def optimize_artificial_fitness(
    rng: PRNGKeyT,
    starting_seq: list[str] | ds.Dataset,
    fitness_func: ArtificialFitness,
    num_rounds: int,
    policy: BatchOptPolicy,
    batch_size: int = 96,
    progress=False,
) -> ds.Dataset:
    """Split data into train and test. Then train, predict and plot predictions as well as prediction quality metrics.

    Args:
        rng: A random number generator key. Defaults to PRNGKey(0).
        starting_seq: List of sequences to start the evolution with, or dataset of measured values
        fitness_func: Artificial fitness function.
        num_rounds: How many rounds of optimization are calculated.
        policy: Decision policy is used to pick the best new sequences.
        batch_size: Number of sequences are picked in each optimization process. (default is 96)
        progress: Flag if progress is shown. (default is False)

    Returns:
        measured: List of all new predicted sequences.
    """
    # compute the fitness of the starting sequence in the in-silico wetlab
    if isinstance(starting_seq, ds.Dataset):
        measured = starting_seq
    else:
        measured = fitness_func(starting_seq)

    # Do we need a progress bar?
    rng_keys = split(rng, num_rounds)
    if not progress:

        def progress_bar(x):
            return x

    else:
        progress_bar = tqdm.tqdm

    # num_rounds iterations of the cycle "Learn -> Design -> Build -> Test"
    for i, round_rng in progress_bar(enumerate(rng_keys)):
        print(f"Batch {i}:")
        # Learn & design through policy
        to_measure = policy(round_rng, measured, batch_size)

        # Build and test through artificial fitness
        tmp = fitness_func(to_measure)
        appended = ds.concatenate_datasets([measured, tmp])

        # add the new sequences to our measured ground truth knowledge
        measured = appended

    return measured
