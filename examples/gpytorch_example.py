from eep.models import GaussianKernelRegression
from pedata.preprocessing import DatasetSplitterRandomTrainTest
from pedata.config import add_encodings
from datasets import load_dataset
from eep.experiment import metrics
from eep.plot import plot_pred
import numpy as np
import os
from pathlib import Path
from jax.random import PRNGKey


def gaussian_k_regr(
    dataset: str,
    feature_name: list[str],
):
    """example of using GaussianKernelRegression
    Args:
        dataset_name: dataset repo_id from huggingface
        feature_name: encoding(s) to use
    Returns:
        dict: results containing the metrics for train and test
    """
    # add encodings
    dataset = add_encodings(dataset, needed=feature_name)

    # get train / test splits - Company class
    dataset = DatasetSplitterRandomTrainTest().split(dataset)

    # instantiating the model
    model = GaussianKernelRegression(
        feature_name, target_standardization=True, epochs=100
    )
    # fitting the model
    model.fit(rng=PRNGKey(0), train_ds=dataset["train"])
    train_prediction = model.predict(test_ds=dataset["train"])
    train_gt = model.get_target(dataset["train"], as_dict=True)
    train_metrics = metrics(train_prediction, train_gt)

    # predicting on the test set
    test_prediction = model.predict(test_ds=dataset["test"])
    test_gt = model.get_target(dataset["test"], as_dict=True)
    test_metrics = metrics(test_prediction, test_gt)

    return {"train": train_metrics, "test": test_metrics}


def plot_results(
    results: dict[str, dict[str, dict[str, np.ndarray]]],
    fig_dir: str | Path,
    regr_name: str,
) -> None:
    """Plot results from gaussian_k_regr
    Args:
        results: output dictionary from gaussian_k_regr"""
    if isinstance(fig_dir, str):
        fig_dir = Path(fig_dir)
    print(f"Plotting results in {fig_dir}")
    for split_name, dataset_split in results.items():
        for target in dataset_split.keys():
            plot_pred(
                ground_truth=dataset_split[target]["gt"],
                colors=np.arange(len(dataset_split[target]["gt"])),
                mean=dataset_split[target]["pred_mean"],
                var=dataset_split[target]["pred_var"],
                target=target,
                filename=os.path.join(fig_dir, f"{regr_name}_plot_{split_name}.png"),
            )

            print(f" --- SPLIT: {split_name} -------")
            print(f" - MAE = {results[split_name][target]['MAE']}")
            print(f" - RMSE = {results[split_name][target]['RMSE']}")
            print(f" - ρ = {results[split_name][target]['ρ']}")
            print(f" - τ = {results[split_name][target]['τ']}")
            print(f" - top100 = {results[split_name][target]['top100']}")


if __name__ == "__main__":
    results_in = Path("examples/results")
    os.makedirs(results_in, exist_ok=True)

    if True:
        print("--- GaussianKernelRegression on Amino Acid dataset ---")
        # getting data
        dataset = load_dataset("Company/CrHydA1_PE_REGR")

        results = gaussian_k_regr(
            dataset=dataset,
            feature_name=["aa_1hot"],
        )
        plot_results(results, fig_dir=results_in, regr_name="aa_1hot_regr")

    if True:
        print("-----------------------------------------------------------------")
        print("--- GaussianKernelRegression on Molecular dataset with smiles ---")
        dataset_name = "Company/Lipophilicity_MOL_REGR"
        print(f"--- Dataset: {dataset_name} ---")

        dataset = (
            load_dataset("Company/Lipophilicity_MOL_REGR")["train"].shuffle(
                generator=np.random.default_rng(12345)
            )
            # .select(list(range(1000)))
        )

        results = gaussian_k_regr(
            dataset=dataset,
            feature_name=["smiles_1hot"],
        )
        plot_results(results, fig_dir=results_in, regr_name=("smiles_1hot_regr"))
