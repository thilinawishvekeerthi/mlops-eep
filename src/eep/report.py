import base64
import os
import io
import sys
from contextlib import contextmanager
from io import StringIO
import datasets as ds
from datasets import load_dataset
from huggingface_hub import HfApi, DatasetFilter
import pandas as pd
import jax
import jax.numpy as np
from jax import random
from jax.random import PRNGKey
import markdown
import matplotlib.pyplot as plt

import pedata as ped
import pedata.config.paths as paths
from pedata.util import append_summary_variable, zscore

import eep.experiment as krr
from eep.models import RFuniRep
from eep.models import GaussianKernelRegression
from eep.plot import plot_pred

jax.config.update("jax_debug_nans", True)


def check_if_dataset_on_hf(dataset_name: str) -> bool:
    """Check if a dataset is available on the Company space on HuggingFace

    Args:
        dataset_name: Name of the dataset.

    Returns:
        True if the dataset is available, False otherwise.
    """
    if not isinstance(dataset_name, str):
        raise Warning(
            f"dataset_name is here {type(dataset_name)} but should be a string"
        )

    api = HfApi()
    author_filter = DatasetFilter(author="Company")
    datasets_Company = api.list_datasets(filter=author_filter)
    list_datasets_Company = sorted([dataset.id for dataset in datasets_Company])
    for dataset in list_datasets_Company:
        if dataset_name == dataset:
            return True
        else:
            continue

    return False


def aggregate_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average performance over all targets and all methods.

    Args:
        df (pd.DataFrame): Dataframe with results from `get_cv_results`.

    Returns:
        pd.DataFrame: Dataframe with average performance over all targets and all methods.
    """
    return (
        df.drop(columns=["cv_split", "target", "top100", "gt"])[
            ["MAE", "RMSE", "ρ", "τ", "method"]
        ]
        .groupby(["method"])
        .mean()
    )


@contextmanager
def suppress_stdout():
    """Suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stderr = sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def get_cv_results(
    nruns: int, df: ds.Dataset, train_frac: float = 0.8, rngkey: PRNGKey = PRNGKey(0)
) -> tuple[pd.DataFrame, dict[str, any], bool]:
    """Compute cross-validation results for a selection of prediction models.

    Args:
        nruns: Number of cross-validation runs.
        df: Dataset to use for cross-validation.
        train_frac: Fraction of data to use for training. Defaults to 0.8.
        rngkey: Random number generator key. Defaults to PRNGKey(0).

    Returns:
        Results for individual CV runs, dictionary with aggregated results and a boolean indicating whether a good regressor was found.
    """
    with suppress_stdout():

        def init_noise(rng: PRNGKey, shape: tuple[int] = (1,)) -> np.array:
            """Random initialization noise

            Args:
                rng: Random number generator key.
                shape: Shape of the noise. Defaults to (1,).

            Returns:
                Array: Initialization noise.
            """
            return random.normal(rng, shape) / 10 * 0

    def get_res(
        rngkey: PRNGKey,
        split_no: int,
        dataset: ds.Dataset,
        rval: list[dict[str, dict[str, any]]] = [],
        train_frac: float = 0.8,
        optimism: float = 0.5,
    ) -> None:
        """Compute cross-validation results for a single split.

        Args:
            rngkey: Random rngkey. This determines the split of the data.x
            split_no: Index number of the current split.
            dataset: The dataset to use for cross-validation.
            rval: List of results that will be appended. Defaults to [].
            train_frac: Fraction of data to use for training. Defaults to 0.8.
            optimism: Optimism parameter for the batch UCB policy. Defaults to 0.5.

        Raises:
            TypeError: If dataset is not a Dataset object.
        """
        if not isinstance(dataset, ds.Dataset):
            raise TypeError(
                f"dataset must be a Dataset object and it is here a {type(dataset)}"
            )

        # instantiate the gaussian kernel regression model
        gr = GaussianKernelRegression(
            ["aa_1hot"], target_standardization=True, epochs=50
        )
        # instantiate the random forest model
        rf = RFuniRep(True)
        with suppress_stdout():
            res_gr = krr.check_predictions(
                dataset,
                gr,
                rng=rngkey,
                ndata=2000,
                train_frac=train_frac,
                optimism=optimism,
            )
            res_rf = krr.check_predictions(
                dataset,
                rf,
                rng=rngkey,
                ndata=2000,
                train_frac=train_frac,
                optimism=optimism,
            )
            for n, r in [("Model1", res_rf), ("Model2", res_gr)]:
                for k in r["test"]:
                    rval.append(r["test"][k])
                    rval[-1]["cv_split"] = split_no
                    rval[-1]["method"] = n
                    rval[-1]["target"] = (
                        k.lstrip("target").lstrip().lstrip("_").lstrip()
                    )

    rval = []
    key_run = random.split(rngkey, nruns)
    for i in range(nruns):
        with suppress_stdout():
            get_res(key_run[i], i, df, rval, train_frac)
    df = pd.DataFrame(rval)
    avg_perf = aggregate_performance(df)

    good_regressor = False
    for _, method in avg_perf.iterrows():
        if method["ρ"] > 0.4 and method["τ"] > 0.3:
            good_regressor = True
            break
    return rval, avg_perf, good_regressor


def generate_plots(
    model_name: str, num_plots: int, cv_run: tuple[pd.DataFrame, dict[str, any], bool]
) -> dict[str, dict[str, any]]:
    """Generate plots for a given model and cross-validation run.

    Args:
        model_name: Name of the model.
        num_plots: Number of plots to generate.
        cv_run: The results of the cross validation fold.

    Returns:
        Nested dictionary with plots.
    """
    plt.style.use("seaborn-v0_8-poster")

    df = pd.DataFrame(cv_run)
    model_df = df[df.method == model_name]
    gb = model_df.groupby(["target"])
    plots = {}

    for targ in gb.groups.keys():
        plots[targ] = {"prioritization": [], "accuracy": []}
        for prio_resc in (True, False):
            prio_resc_name = "prioritization" if prio_resc else "accuracy"
            style = "dec_crit" if prio_resc else "uncertainty"
            i = -1
            while len(plots[targ][prio_resc_name]) < num_plots and i < len(cv_run) - 1:
                i += 1
                if df.iloc[i]["method"] != model_name or df.iloc[i]["target"] != targ:
                    continue
                p = plot_pred(
                    df.iloc[i]["gt"],
                    np.arange(df.iloc[i]["gt"].size),
                    df.iloc[i]["pred_mean"],
                    df.iloc[i]["pred_var"],
                    show=False,
                    target=targ,
                    style=style,
                )  # prioritization_rescale = prio_resc)
                p.tight_layout()
                plots[targ][prio_resc_name].append(p)
    return plots


def get_html_img_from_fig(fig, file_format: str = "png", css_class: str = None) -> None:
    """Generate base64-encoded inline HTML image from a matplotlib figure.

    Args:
        fig: Matplotlib figure.
        file_format: File name extension of the image. Defaults to "png".
        css_class: CSS class to apply to the image. Defaults to None, in which case no class is applied.
    """

    f = io.BytesIO()
    fig.savefig(f, format=file_format)
    rval = f.getvalue()
    f.close()

    if css_class is not None:
        css_class = f"class='{css_class}' "
    rval = (
        "<img "
        + css_class
        + f"src='data:image/{file_format};base64,"
        + base64.b64encode(rval).decode("ASCII")
        + "'/>"
    )
    return rval


def get_html_plots_one_target(plot_targ: dict[str, any]) -> str:
    """Generate HTML code for all plots of a single target.

    Args:
        plot_targ: Dictionary with matplotlib plots for a single target. Expected to have keys "prioritization" and "accuracy".

    Returns:
        HTML code for the plots.
    """
    rval = "<table class='plotsTable'>\n"
    rval += "<thead><tr><th>CV iteration</th><th>Prediction accuracy</th><th>Prioritization visualization</th></tr></thead>"
    rval += "<tbody>"

    for i in range(len(plot_targ["prioritization"])):
        rval += f"""
    <tr>
        <td>Iteration {i+1}</td>
        <td>{get_html_img_from_fig(plot_targ['accuracy'][i], 'png', 'plot')}</td>
        <td>{get_html_img_from_fig(plot_targ['prioritization'][i], 'png', 'plot')}</td>
    </tr>"""

    rval += "</tbody>"
    rval += "\n</table>"
    return rval


def get_html_plots_all_targets(plots: dict[str, dict[str, any]]) -> str:
    """Generate HTML code for all plots of all targets.

    Args:
        plots (dict[str, dict[str, any]]): Nested dictionary with matplotlib plots for all targets. Expected to have keys for each target, and each target to have keys "prioritization" and "accuracy".

    Returns:
        str: HTML code for the plots.
    """
    if len(plots) == 1:
        return get_html_plots_one_target(list(plots.values())[0])
    else:
        rval = '<div class="tab">'
        first = True
        for targ in plots:
            rval += f"<button class='tablinks' onclick='openTab(event, \"{targ}\")'"
            if first:
                rval += ' id="defaultOpen"'
                first = False
            rval += f")'>{targ}</button>"
        rval += "</div>"

        for targ in plots:
            rval += f'<div id="{targ}" class="tabcontent">'
            rval += get_html_plots_one_target(plots[targ])
            rval += "</div>"

        rval += '<script>document.getElementById("defaultOpen").click(); </script>'
        return rval


def get_report(cv_run: pd.DataFrame, avg_perf: dict[str, any], dataname: str) -> str:
    """Generate HTML report for a cross-validation run.

    Args:
        cv_run (pd.DataFrame): Cross-validation runs.
        avg_perf (dict[str, any]): Average performance of the models.
        dataname (str): Name of the dataset.

    Returns:
        str: HTML code for the report.
    """

    recommended = False
    target_names = pd.DataFrame(cv_run).target.unique().tolist()

    best_model = avg_perf.index[avg_perf.iloc[:, 2:].mean(1).argmax()]
    targ_grouped = (
        pd.DataFrame(cv_run)[["target", "ρ", "τ", "method"]]
        .groupby(["target", "method"])
        .mean()
    )

    for idx, method in avg_perf.iterrows():
        if method["ρ"] > 0.4 and method["τ"] > 0.3:
            recommended = True
            break

    if recommended:
        criterion = f"Spearman's &rho; &GreaterEqual; 0.4 and Kendall's &tau; &GreaterEqual; 0.3 (for {method.name})"
        recommendation = True  # FIXME is this necessary?
        rec = "recommend using our prediction and suggestion algorithms"
    else:
        criterion = f"Spearman's &rho; < 0.4 and Kendall's &tau; < 0.3 for all models"
        recommendation = False  # FIXME is this necessary?
        rec = "can not recommend using our prediction and suggestion algorithms using the current data set"

    plots = generate_plots(best_model, 3, cv_run)
    assert len(plots) > 0

    num_cv_folds = pd.DataFrame(cv_run).cv_split.unique().size

    md = markdown.Markdown(extensions=["toc"])

    var = f"""![](https://Company.com/images/logo.svg)
# Report on _{dataname}_

The provided dataset contains protein sequences and measurements of associated _{", ".join(target_names)}_.

We **{rec}**.
Our recommendation is based on the fact that {criterion} for rank correlation between prediction and measured values.
The results of averaging {num_cv_folds} iterations with different training/test splits are reported in the table below.
    """
    perf_table = avg_perf.loc[:, ["ρ", "τ"]].to_html(
        float_format="%.2f", index_names=False
    )
    perf_table = perf_table.replace("ρ", "&rho;").replace("τ", "&tau;")
    var_html = (
        md.convert(var)
        + perf_table
        + md.convert(
            "To provide an overall assessment of the performance of our prediction, rank correlation was averaged over all targets. For rank correlations for each target and model see below. If you have not read one of our feasibility reports before, please refer to the [background section](#background) for understanding some of the statistical language used."
        )
    )

    if len(plots) == 1:
        table_target = ""
    else:
        table_target = "The rank correlations per target reported in the following table reveal potential differences in predicting each target. They provide a quantitative assessment of the predictions."
        table_target += (
            targ_grouped.to_html(float_format="%.2f", index_names=False)
            .replace("ρ", "&rho;")
            .replace("τ", "&tau;")
        )

    pred_prio_html = md.convert(
        f"""## Results for individual target properties
In this section, we report the results of our prediction algorithm for each target property. {table_target}"""
    )

    pred_prio_html += md.convert(
        f"""We visualize prediction accuracy and prioritization for each target.
    The plots are generated with {best_model}. They provide a qualitative impression of the predictions. We provide several plots, each resulting from the test set of a different [cross validation iteration](#cross-validation). This way, you can get a better impression of how stable the results are and how reproducible they will be when actually using the model for prioritizing experiments.

**Prediction accuracy plots (left)** include ground truth, predicted mean and uncertainty of predictions (&sigma;). The predicted mean is the best guess of the model. The prediction uncertainty is low if the model has seen similar variants in the training data, higher if this was not the case.

**Prioritization visualizations (right)** shows only the decision trend based on the prediction. If prioritization works as expected (&rho; and &tau; > 0), the decision trend increases from left to right. Higher slopes mean better decisions."""
    )

    pred_prio_html += get_html_plots_all_targets(plots)

    explanation1 = md.convert(
        f"""
## Background

In this section, we give some statistical background on the reported metrics and provided plots. If you have read an Company feasibility report before or you know statistics well, feel free to skip this section.

### Rank correlations
We quantitatively assess how well Companys prediction models would help to prioritize experiments given the current data set (and its measurement noise). For this, we report rank correlation between prediction and actual measurements in terms of [Spearman's &rho;](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) and [Kendall's &tau;](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient). This means training our prediction algorithm only with 80% of the variants (training data), predicting functional measurement for the other 20% in the variants (test data), and computing rank correlation between prediction and actual measurements.  The larger rank correlation is, the better prioritization works. The highest possible value is 1 (perfect rank correlation), the lowest is -1 (perfect anticorrelation of ranks) for both &rho; and &tau;.

### Cross validation
We use [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) (CV) to compute reliable numbers for rank correlations.
From a biochemists perspective, cross validation tries to mimic the way experiments are done based on data.
In other words, some sequence variants have already been synthesized and their properties measured, and from these measurements we try to guess how new variants might behave.

We achive this by showing only some of the data you provided to our prediction algorithm for it to learn from (the training data).
Then we ask it to predict the properties for the sequences that it has not seen previously (the test data).

This mimics what you hope to do: given measurements for some sequences, predict the measurements of new sequences, and use the prediction to prioritize new sequences.
The following visualization shows how we can repeat this procedure several times.
"""
    )
    cvVis = """<table cellspacing="0" cellpadding="0" class="cvVis">
                    <tbody>
                        <tr style="height: 20px">
                            <td class="s0" dir="ltr">Iteration</td>
                            <td class="s1" dir="ltr">1</td>
                            <td class="sTrain" dir="ltr">Training</td>
                            <td class="sTrain"></td>
                            <td class="sTrain"></td>
                            <td class="sTest" dir="ltr">Test</td>
                            <td class="s2" dir="ltr">&rarr; &tau;1, &rho;1</td>
                        </tr>
                        <tr style="height: 20px">
                            <td class="s0" dir="ltr">Iteration</td>
                            <td class="s1" dir="ltr">2</td>
                            <td class="sTrain" dir="ltr">Training</td>
                            <td class="sTrain"></td>
                            <td class="s2" dir="ltr">Test</td>
                            <td class="sTrain" dir="ltr">Training</td>
                            <td class="s2" dir="ltr">&rarr; &tau;2, &rho;2</td>
                        </tr>
                        <tr style="height: 20px">
                            <td class="s0" dir="ltr">Iteration</td>
                            <td class="s1" dir="ltr">3</td>
                            <td class="sTrain" dir="ltr">Training</td>
                            <td class="s2" dir="ltr">Test</td>
                            <td class="sTrain" dir="ltr">Training</td>
                            <td class="sTrain"></td>
                            <td class="s2" dir="ltr">&rarr; &tau;3, &rho;3</td>
                        </tr>
                        <tr style="height: 20px">
                            <td class="s0" dir="ltr">Iteration</td>
                            <td class="s1" dir="ltr">4</td>
                            <td class="sTest" dir="ltr">Test</td>
                            <td class="sTrain" dir="ltr">Training</td>
                            <td class="sTrain"></td>
                            <td class="sTrain"></td>
                            <td class="s2" dir="ltr">&rarr; &tau;4, &rho;4</td>
                        </tr>
                    </tbody>
                </table>
    """

    explanation2 = md.convert(
        f"""In this example, we divide the data into 4 equally large parts, each part containing roughly the same number of sequences.
In the first iteration, we train the algorithm on 3 of those parts and ask it to predict measurements for the 4th.
In the next iterations this is repeated, only that the parts of the data used for training and testing are changed.
The number of iterations is called _folds_ in cross validation.
For the report you are currently reading, we used {num_cv_folds} folds.

This results in several rank correlation numbers, the average of which is given in the section [Rank correlation results](#rank-correlation-results).
Example prediction and prioritization plots for the different test sets resulting from fold are also [provided above](#example-prediction-and-prioritization-plots)."""
    )

    explanation = explanation1 + cvVis + explanation2

    theme_bg_color_dark = "#339604"
    f = StringIO()
    f.write(
        """
<!DOCTYPE html>
<html>
<head>
<title>Report on """
        + dataname
        + """</title>
<meta charset="8859-7">
<style>

body {font-family: Arial, Helvetica, sans-serif;}

:root {
    --Company-dark-green:#339604;
    --Company-light-green: #70CB02;
    --dark-grey: #888;
    --light-grey: #ddd;
}

.plotsTable {
width: 100%;
}

.cvVis, .cvVis tr, .cvVis td {
border: 0px;
background-color: #ffffff;
break-inside: avoid;
}

.cvVis .sTrain {
background-color: var(--Company-dark-green);
color: #ffffff;
text-align: center;
border: 1px solid black;
}

.cvVis .sTest {
background-color: #ffffff;
text-align: center;
border: 1px solid black;
}

table {
font-family: Arial, Helvetica, sans-serif;
border-collapse: collapse;
border-top: none;
text-align: center;
}

td, th {
border: 1px solid var(--light-grey);
padding: 8px;
}

tr:nth-child(even){background-color: #f2f2f2;}

tr:hover {background-color: var(--light-grey);}

th {
padding-top: 12px;
padding-bottom: 12px;
text-align: left;
}

thead th {
background-color: var(--Company-dark-green);
color: #ffffff;
}


img.plot {
width:100%
}

/*Tab styles*/
.tab {
  overflow: hidden;
}

/* Style the buttons that are used to open the tab content */
.tab button {
  position: relative;
  font-size:large;
  z-index: 2;
  top: 2px;
  margin: 0;
  margin-top: 4px;
  padding: 6px 6px 8px;
  border: 1px solid hsl(219deg 1% 72%);
  border-radius: 5px 5px 0 0;
  overflow: visible;
  background: var(--light-grey);
  outline: none;
  color: rgb(40 40 40);
  font-weight: bold;
  transition: 0.3s;

}

/* Change background color of buttons on hover
.tab button:hover {
  background-color: var(--light-grey);
}*/

/* Create an active/current tablink class */
.tab button.active {
  padding: 5px 5px 7px;
  margin-top: 0;
  z-index: 3;
  border-top-color: var(--dark-grey);
  border-top-color: var(--Company-dark-green);
  border-width: 2px;
  border-top-width: 6px;
  border-bottom: none;
  color: black;
  background: white;
  box-shadow: 2px 2px 5px  var(--dark-grey) ;
}

/* Style the tab content */
.tabcontent {
  display: none;
  padding: 0px 0px;
  border: none; /*1px solid var(--Company-dark-green);*/
}
</style>

<script>
function openTab(evt, cityName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(cityName).style.display = "block";
  evt.currentTarget.className += " active";
}
</script>

</head>
<body>
"""
    )
    f.write(
        f"""{var_html + pred_prio_html + explanation}
</body>
</html>"""
    )
    rval = f.getvalue()
    f.close()
    return rval


if __name__ == "__main__":
    import pickle
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--nfolds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "-d",
        "--dataname",
        type=str,
        default="Unnamed dataset",
        help="Name of the dataset",
    )

    parser.add_argument(
        "-t",
        "--training",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (between 0 and 1)",
    )

    parser.add_argument(
        "datafile",
        help="Path to the data file or name of the dataset in standard datasets directory",
    )
    parser.add_argument(
        "-o", "--output", type=argparse.FileType("w"), default=None, help="Output file"
    )

    p = parser.parse_args()
    if p.output is None:
        p.output = open("Report_" + p.dataname + ".html", "w")

    if os.path.exists(p.datafile):
        df = ped.preprocess_data(p.datafile)
    elif os.path.exists(paths.PE_DATA_DIR / p.datafile):
        df = ds.load_from_disk(paths.PE_DATA_DIR / p.datafile)
    elif check_if_dataset_on_hf(p.datafile):
        df = load_dataset(p.datafile)["whole_dataset"]
    else:
        print(check_if_dataset_on_hf(p.datafile))
        raise ValueError(f"Data file '{p.datafile}' not found")

    # FIXME: Objectives is set to the default value (objectives=None) ; We need to add a parameter to the command line
    # OptimizationObjective

    df = append_summary_variable(
        dataset=df,
        normalization=zscore,
        objectives=None,
        summary_variable_name="target summary variable",
    )
    cv_runs, avg_perf, good_performance = get_cv_results(
        p.nfolds, df, train_frac=p.training
    )

    with open(f"{p.dataname}_cv{p.nfolds}.pkl", "wb") as f:
        pickle.dump([cv_runs, avg_perf, p.dataname], f)

    rep_html = get_report(cv_runs, avg_perf, p.dataname)

    with p.output as f:
        f.write(rep_html)
