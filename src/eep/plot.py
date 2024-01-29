from typing import Tuple
import matplotlib.pyplot as plt
import jax.numpy as np
import pandas as pd
import numpy as onp
import seaborn as sb

sb.set()
pal = onp.array(sb.color_palette())


def plot_pred(
    ground_truth: np.ndarray,
    colors: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray = None,
    filename: str = None,
    style: str = "uncertainty",
    target: str = None,
    show: bool = False,
    optimism: float = 0.5,
    prioritization_rescale: bool = False,
    figsize=(8, 4),
):
    """Visualizes the prediction against the true values.

    Args:
        ground_truth (np.ndarray): True value of the predicted data.
        colors (np.ndarray): Colors used be the plotting routine.
        mean (np.ndarray): Predicted mean value.
        var (np.ndarray): Predicted variance for the dataset. Defaults to None.
        filename (str): Filename for saving the plot.
        style (str): Which style should be used for the plot.
        target (str): Name of the target variable (default is None)
        show (bool): Flag if the plot should be displayed.
        optimism (float): Optimism parameter for decision criterion. Defaults to 0.5.
        prioritization_rescale (bool): Whether to rescale for better visualization of prioritization

    Returns:
        fig (matplotlib.figure.Figure): Figure object.

    Raises:
        ValueError: If target is not given.
    """

    # style="dec_crit"

    if target is None:
        raise ValueError("target must be given")

    if style is None:
        return

    sd = None
    predlabel = "Prediction"
    order = onp.argsort(ground_truth)
    mean_bias, mean_sd = mean.mean(), mean.std()
    gt_bias, gt_sd = ground_truth.mean(), ground_truth.std()
    if prioritization_rescale:
        mean = (mean - mean_bias) / mean_sd
        ground_truth = (ground_truth - gt_bias) / gt_sd
    if var is None:
        dec_crit = None
    else:
        predlabel = "Predicted mean (±σ, ±2σ)"
        sd = np.sqrt(np.clip(var, 0.0000))
        if prioritization_rescale:
            sd = sd / gt_sd
        dec_crit = mean + optimism * sd
        # order = np.argsort(dec_crit+ground_truth)
        dec_crit = dec_crit[order]
    x_pos = np.arange(mean.size)
    fig, ax = plt.subplots(figsize=figsize)

    mean = mean[order]
    if var is not None:
        sd = sd[order]
    gt = ground_truth[order]
    colors = colors.ravel()[onp.array(order)]
    predcol = "#339604"
    gtcol = "#666666"

    ax.set_xticks([])
    handles = []
    if style == "bars":
        ax.bar(
            x_pos, mean, yerr=sd, align="center", alpha=0.5, ecolor="black", capsize=10
        )
        ax.set_ylabel("Predicted: bar, actual: dot")
        ax.scatter(x_pos, gt)
        ax.set_ylabel(target)
        ax.legend(handles=handles, loc="best")
    else:
        ax.set_xticklabels([])
        if style == "point_est":
            handles.append(
                ax.scatter(x_pos, gt, c=pal[0 * colors], label=f"Ground truth")
            )
            handles.append(
                ax.scatter(x_pos, mean, marker="+", color=predcol, label=predlabel)
            )
            ax.set_ylabel(target)
            ax.legend(handles=handles, loc="best")
        elif style == "dec_crit":
            assert dec_crit is not None

            dec_model = onp.polyfit(x_pos.astype(np.float32), dec_crit, 1)
            gt_model = onp.polyfit(x_pos.astype(np.float32), gt, 1)
            # dec_trend_model = gt_model * onp.array([dec_model[0], 1.])
            dec_trend_model = onp.array([dec_model[0], 0.0])
            # handles.append(ax.scatter(x_pos, (dec_crit - dec_crit.mean()) / dec_crit.std(), marker="x", color=predcol, label="decision criterion", alpha=0.8))
            dec_trend = onp.poly1d(dec_trend_model)(x_pos.astype(np.float32))
            dec_intercept = onp.poly1d(gt_model)(x_pos.mean()) - onp.poly1d(
                dec_trend_model
            )(x_pos.mean())
            handles.append(
                ax.scatter(
                    x_pos,
                    gt,
                    color=gtcol,
                    marker="o",
                    facecolor="white",
                    label=f"Ground truth",
                )
            )
            handles.append(
                ax.plot(
                    x_pos,
                    dec_trend + dec_intercept,
                    color=predcol,
                    label="priorization trend",
                    alpha=0.8,
                )
            )
            ax.set_xticklabels([])
            ax.set_ylabel(target)
            ax.legend(loc="best")
        else:
            if sd is not None:
                # assert False
                # ax.scatter(np.hstack([x_pos]*2), np.hstack([l, u]), marker="_", color=predcol)
                for mult in range(1, 3):
                    m = mult
                    ax.fill_between(
                        x_pos,
                        mean - m * sd,
                        mean + m * sd,
                        color=predcol,
                        linewidth=0.0,
                        alpha=0.4 - mult * 0.1,
                    )
            handles.append(
                ax.scatter(
                    x_pos, mean, marker="+", s=70, color=predcol, label=predlabel
                )
            )
            handles.append(
                ax.scatter(
                    x_pos,
                    gt,
                    color=gtcol,
                    marker="o",
                    facecolor="white",
                    label="Ground truth",
                    alpha=0.8,
                )
            )
            ax.set_ylabel(target)
            ax.legend(handles=handles, loc="best")

    # ax.set_title('activity µmol H2 / (µg * min)')
    ax.set_xlabel(f"{mean.size} variants (ordered by ground truth)")
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=300)
    if show:
        fig.show()
    else:
        plt.close()
    return fig


def best_found_per_batch(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """Returns the best sequence per methods and batch and puts them into a new dataframe.

    Args:
        df (pd.DataFrame): Dataframe for search in.
        batch_size (int): Size of the batch.

    Returns:
        rval (pd.DatFrame): DataFrame with best sequences
    """
    rval = pd.DataFrame({"method": [], "batch": [], "max_val": []})
    for name, group in df.groupby("method"):
        batch = []
        max_val = []
        for i in range(batch_size, len(group), batch_size):
            batch.append(i)
            max_val.append(group["target summary variable"].values[:i].max())
        c = pd.DataFrame({"batch": batch, "max_val": max_val})
        c["method"] = name
        rval = rval.append(c)
    rval.method = rval.method.astype(pd.CategoricalDtype())
    return rval


def plot_evolution(
    df: pd.DataFrame,
    batch_size: int,
    filename: str,
    figsize: Tuple[float, float] = (5.0, 3.5),
    seq_cost: float = None,
) -> None:
    """Visualizes the gain of the evolution.

    Args:
        df (pd.DataFrame): Dataframe for search in.
        batch_size (int): Size of the batch.
        filename (str): Name for saving the figure.
        figsize (Tuple[float, float]): Size of resulting figure. Defaults to `(5., 3.5)`.
        seq_cost (float): Cost of a sequence in USD. If given, plot protein fitness vs. development cost. Defaults to `None`, in which case protein fitness vs. # of experimental sequences is plotted

    Returns:
        None
    """
    best = best_found_per_batch(df, batch_size)
    fig, ax = plt.subplots(figsize=figsize)
    plt.tight_layout()
    for m in best.method.cat.categories:
        print(m)
        label = m
        style = "-"
        if m.startswith("AI"):
            if "unbatched" in m:
                style = ":"
                label = "AI (unbatched)"
            else:
                label = "Company AI"
            col = "#339604"
        else:
            label = "Directed Evolution"
            col = "#666666"
        current = best[best.method == m]
        if seq_cost is None:
            if "unbatched" not in m:
                ax.plot(
                    current.batch,
                    current.max_val,
                    style,
                    label=label,
                    color=col,
                    linewidth=2,
                )
        else:
            ax.plot(
                current.batch * seq_cost / 1000,
                current.max_val,
                style,
                label=label,
                color=col,
                linewidth=2,
            )
    if seq_cost is None:
        ax.set_xlabel("total # of enzyme variants")
        ax.set_ylabel("Protein fitness")
        ax.set_title(f"Best found variant (batchsize {batch_size})")
    else:
        ax.set_xlabel("Experimental cost (in thousand USD)")
        ax.set_ylabel("Protein fitness")
        # ax.set_title(f"Cost for sequence development (batchsize {batch_size})")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
