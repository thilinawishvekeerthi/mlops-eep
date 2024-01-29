from pytest import fixture
from datasets import load_dataset
import numpy as np
from jax.random import PRNGKey
import pandas as pd
from eep.models import GaussianKernelRegression, CustomGPModel
from eep import plot
import seaborn as sb
import os


# fixture for regr_dataset_train and regr_dataset_test
@fixture(scope="module")
def needed_encodings():
    return ["aa_1hot"]


# ========= figure and summary folder ---
if not os.path.exists("code_health/summaries_and_plots"):
    os.makedirs("code_health/summaries_and_plots", exist_ok=True)

# ========= Colors for plots -------
sb.set()
color_pal = np.array(sb.color_palette())
plot_target = False

# ========= FIXTURES ----------------
# are in test/conftest.py


# ========= helper functions --------
# testing kernel base function
def train_and_eval_model(
    ModelClass, regr_dataset_train, regr_dataset_test, epochs=50, plot_predictions=False
):
    """Test to see if the model runs."""
    features = ["aa_1hot"]
    model = GaussianKernelRegression(
        feature_list=features,
        target_standardization=True,
        epochs=epochs,
        modelClass=ModelClass,
    )
    # does training pass?
    model.fit(rng=PRNGKey(0), train_ds=regr_dataset_train)
    # does prediction work?
    test_prediction = model.predict(test_ds=regr_dataset_test)
    target_name, _ = model.get_target(regr_dataset_train)

    summary = {
        "model": model,
        "features": features,
        "epochs_nb": epochs,
        "saving_filename": f"{model.ModelClass.__name__}_epochs{epochs}",
        "target_name": target_name,
    }

    summary["results"] = pd.DataFrame(
        {
            target_name[0]: regr_dataset_test[target_name[0]],
            "prediction_mean": test_prediction[target_name[0]]["mean"],
            "prediction_var": test_prediction[target_name[0]]["var"],
        }
    )

    # nans in the test_prediction?
    assert not summary["results"].loc[:, "prediction_mean"].isna().any()
    assert not summary["results"].loc[:, "prediction_var"].isna().any()
    assert all(
        test_prediction[target_name[0]]["mean"].round(1)
        == np.array([1744.3, 2479.4, 1744.3, 1863.0, 1254.7])
    )

    # plot the predictions
    if plot_predictions:
        plot_summary(summary)

    return summary


# plotting the prediction and ground truth
def plot_summary(summary):
    plot.plot_pred(
        summary["results"].loc[:, summary["target_name"][0]].to_numpy(),
        color_pal,
        summary["results"].loc[:, "prediction_mean"].to_numpy(),
        summary["results"].loc[:, "prediction_var"].to_numpy(),
        target="target",
        filename=f"code_health/summaries_and_plots/{summary['saving_filename']}.png",
    )


# ==================== Testing if the model trains and predict ====================
def test_GaussianKernelRegression(regr_dataset_train, regr_dataset_test):
    print("=====================================================================")
    train_and_eval_model(
        CustomGPModel,
        regr_dataset_train,
        regr_dataset_test,
        plot_predictions=True,
    )


# ==================== Testing if the model trains and predict ====================
# def test_if_ElementwiseGKRegression_runs(dataset_train, dataset_test):
#     print("=====================================================================")
#     train_and_eval_model(
#         GPModelElementwiseKernel, dataset_train, dataset_test, plot_predictions=True
#     )


# def test_if_ElementwiseNPosGKRegression_runs(dataset_train, dataset_test):
#     print("=====================================================================")
#     train_and_eval_model(
#         GPModelElementwiseNPositionKernel,
#         dataset_train,
#         dataset_test,
#         plot_predictions=True,
#     )


# TODO ==================== Testing how well trained is the model ====================
# def test_GaussianKernelRegression_predictions(dataset_train, dataset_test):
#     print("=====================================================================")
#     summary = train_and_eval_model(CustomGPModel, dataset_train, dataset_test)
#     assess_prediction_metrics(summary)


# TODO
# def test_ElementwiseGKRegression_predictions(dataset_train, dataset_test):
#     print("=====================================================================")
#     summary = train_and_eval_model(
#         GPModelElementwiseKernel, dataset_train, dataset_test
#     )
#     assess_prediction_metrics(summary)


# TODO
# def test_ElementwiseNPosGKRegression_predictions(dataset_train, dataset_test):
#     print("=====================================================================")
#     summary = train_and_eval_model(
#         GPModelElementwiseNPositionKernel, dataset_train, dataset_test
#     )
#     assess_prediction_metrics(summary)


if __name__ == "__main__":
    pass
