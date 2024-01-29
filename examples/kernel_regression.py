from datasets import load_dataset
from eep.models import KernelRegression
from eep.plot import plot_pred
import numpy as np
from pedata.util import get_target

dataset_name = "Company/CrHydA1_PE_REGR"
dataset = load_dataset(dataset_name)["train"]
dataset = dataset.train_test_split(0.2, seed=42)

# kernel regression example
KR = KernelRegression(feature_list=["aa_1hot"], target_standardization=True)
KR.fit(rng=0, train_ds=dataset["train"])
y_pred = KR.predict(test_ds=dataset["test"])

target_name, ground_truth = get_target(dataset["test"], as_dict=False)
y_pred_mean = y_pred[target_name[0]]["mean"].numpy()
fig = plot_pred(
    ground_truth=ground_truth,
    colors=np.arange(len(dataset["test"])),
    mean=y_pred_mean,
    style="uncertainty",
    target=target_name[0],
    show=True,
    filename="examples/KernelRegression_predictions.png",
    figsize=(8, 4),
)
