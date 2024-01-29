import datasets as ds
import jax.numpy as np
from jax.random import PRNGKey
import pedata.mutation as pedm
from pedata.util import (
    OptimizationObjective,
    append_summary_variable,
    get_target_columns,
    zscore,
)
from ..util import PRNGKeyT
import eep.generate as gen
from eep.models import SeqPropertyPred, GaussianKernelRegression
from eep.policies import IndepUCB


def get_parent_and_offset(df: ds.Dataset) -> tuple[str, int]:
    """Get the parent sequence and mutation code offset from the dataset.

    Args:
        df (ds.Dataset): The dataset containing wet lab measurements.

    Returns:
        tuple[str, int]: The parent sequence and mutation code offset.
    """
    parent = pedm.mutation.Mutation.get_parent_aa_seq(df)
    offset = pedm.mutation.Mutation.estimate_offset(df, parent)
    assert offset["matching_ratio"][0] == 1.0, ValueError(
        "Could not reliably guess offset of mutation encodings"
    )
    offset = offset["offset"][0]
    return parent, offset


def get_suggestions(
    df: ds.Dataset,
    total_no_suggestions: int = 32,
    exploration_factor: float = 0.5,
    generator: gen.Generator = gen.RandomGenerator(
        gen.ancestor.BestChoice(), "aa", 1000, 2
    ),
    objectives: dict[str, OptimizationObjective] | None = None,
    random_seed: PRNGKeyT = PRNGKey(0),
    pred_model: SeqPropertyPred | None = None,
) -> ds.Dataset:
    """Get suggestions for the next batch of wet lab experiments.

    Args:
        df (ds.Dataset): The dataset containing wet lab measurements.
        total_no_suggestions (int, optional): The total number of suggestions to return. Defaults to None, returning all candidates.
        exploration_factor (float, optional): Interpolation factor between 0 and 1 for exploration vs. exploitation. Defaults to 0.5.
        generator (gen.Generator, optional): The candidate generator method. Defaults to gen.RandomGenerator(gen.ancestor.BestChoice(), "aa", 1000, 2).
        objectives (dict[str, tuple[str, float]], optional): Objectives for each target column. Defaults to None, in which case all targets are maximized.
        random_seed (PRNGKeyT, optional): Random number generation key. Defaults to PRNGKey(0).

    Returns:
        ds.Dataset: The suggested batch of candidates, including the predicted values for each target column, decision criterion, and the candidates themselves.
    """
    parent, offset = get_parent_and_offset(df)

    generator.offset = offset

    # no need to check validity – this has been checked at upload already
    df = df.remove_columns("aa_seq").add_column(
        "aa_seq",
        pedm.mutation.Mutation.apply_all_mutations(
            df, offset=offset, check_validity=False
        ),
    )

    # define objectives
    if objectives is None:
        objectives = {
            k: OptimizationObjective(direction="max", weight=1.0)
            for k in get_target_columns(df)
        }

    # add summary variable
    df = append_summary_variable(
        df,
        normalization=zscore,
        objectives=objectives,
        summary_variable_name="target summary variable",
    )

    # model
    if pred_model is None:
        pred_model = GaussianKernelRegression(
            ["aa_1hot"], target_standardization=True, epochs=50
        )

    # policy
    policy = IndepUCB(
        pred_model,
        exploration_factor=exploration_factor,
        generator=generator,
    )

    # get suggestions
    suggestions = policy(random_seed, df, total_no_suggestions)

    assert (
        "target summary variable mean" in suggestions.features
        and "target summary variable var" in suggestions.features
    ), ValueError("No predictions made")
    suggestions = suggestions.with_format("numpy")
    suggestions = suggestions.add_column(
        "decision criterion",
        (
            suggestions["target summary variable mean"]
            + exploration_factor * np.sqrt(suggestions["target summary variable var"])
        ).tolist(),
    )
    suggestions = suggestions.with_format(None)
    mut = pedm.extract.extract_mutation_str_from_sequences(
        suggestions["aa_seq"], parent, offset=offset
    )
    suggestions = suggestions.add_column("aa_mut", mut)
    for col in ["aa_1gram", "aa_1hot", "aa_len"]:
        try:
            suggestions = suggestions.remove_columns(col)
        except ValueError:
            pass
    return suggestions
