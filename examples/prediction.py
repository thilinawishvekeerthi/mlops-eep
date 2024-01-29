import datasets as ds
from datasets import load_dataset
import jax.numpy as np
import jax.random as random
import pandas as pd
import pedata.config.paths as p
from jax.random import PRNGKey

import matplotlib.pyplot as plt

import eep.experiment as eepexp
import eep.report as r
import eep.plot as plot
from eep.models import RFuniRep, GaussianKernelRegression, KernelRegression


if __name__ == "__main__":
    rng = PRNGKey(0)
    # dataset
    dataset_name = "Company/CrHydA1_PE_REGR"
    dataset = load_dataset(dataset_name)["train"]
    dataset = dataset.train_test_split(0.2, seed=42)

    # Initialize a GP model
    gp = GaussianKernelRegression(["aa_1hot"], target_standardization=True, epochs=50)

    # Initialize a random forrest model including uncertainty estimates
    rf = RFuniRep(True)

    # Initialize a random forrest model including uncertainty estimates
    KR = KernelRegression(["aa_1hot"], target_standardization=False, epochs=50)

    model_to_run = ((gp, "GP"), (rf, "RF"), (KR, "KR"))

    # train models
    for method, name in model_to_run:
        # You could train and plot prediction easily using
        # eepexp.estimate_and_plot(data["train"], data["test"], method)

        # The longer version that follows is more pedagogical
        method.fit(rng, dataset["train"])

        # get the ground truth
        gt = method.get_target(dataset["test"], as_dict=True)

        # get the predictions
        pred = method.predict(dataset["test"])

        # for each target column, plot the ground truth and the prediction
        for k in gt:
            print(f"plotting {k}")
            # get the variance if the model predicts it
            if "var" in pred[k].keys():
                var = pred[k]["var"].ravel()
            else:
                var = None

            fig = plot.plot_pred(
                gt[k].ravel(),
                np.arange(len(dataset["test"])),
                pred[k]["mean"].ravel(),
                var=var,
                target=k,
                show=True,
            )
            # plt.show()
            # plt.close()
