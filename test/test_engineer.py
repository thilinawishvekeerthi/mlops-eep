import datasets as ds
import numpy as np

from pedata.config import add_encodings
from pedata.util import (
    get_target_columns,
    OptimizationObjective,
    append_summary_variable,
    zscore,
)

import eep.engineer as eng
import eep.generate as gen

from eep.models import GaussianKernelRegression
from eep.policies import IndepUCB, BatchThompson
import eep.util.artif_fitness as af
import pytest
from datasets import concatenate_datasets, load_dataset
import jax


### helper functions ###
def get_small_large_df() -> tuple[ds.Dataset, ds.Dataset, str]:
    base_seq = "MAGLYITR"
    large_df = ds.Dataset.from_dict(
        {
            "aa_seq": [
                "MLGLYITR",
                base_seq,
                "MLYLYITR",
            ],
            "target a": [1, 2, 3],
            "aa_mut": ["A2L", "wildtype", "A2L_G3Y"],
        }
    )

    small_df = ds.Dataset.from_dict(
        {
            "aa_seq": [
                "MLGLYITR",
                base_seq,
            ],
            "target a": [
                1,
                2,
            ],
            "aa_mut": ["A2L", "wildtype"],
        }
    )

    large_df = add_encodings(large_df)
    small_df = add_encodings(small_df)
    for df in [large_df, small_df]:
        df = df.with_format(columns=["aa_seq", "aa_mut"])
    return small_df, large_df, base_seq


# ======== Fixtures ==========


@pytest.fixture()
def small_df():
    small_df, _, _ = get_small_large_df()
    return small_df


@pytest.fixture()
def large_df():
    _, large_df, _ = get_small_large_df()
    return large_df


@pytest.fixture()
def base_seq():
    _, _, base_seq = get_small_large_df()
    return base_seq


@pytest.fixture()
def total_suggestions():
    return 10


@pytest.fixture()
def generator_f():
    return gen.FixedGenerator("aa", ["AA", "GCG"])


@pytest.fixture()
def generator_ds():
    _, large_df, _ = get_small_large_df()
    wt = gen.ancestor.NamedChoice(large_df, "wildtype")(None, large_df)["aa_seq"][0]
    return gen.DeepScanGenerator(wt, "aa", [1, 2, 3], True)


@pytest.fixture()
def objectives():
    _, large_df, _ = get_small_large_df()
    return {
        k: OptimizationObjective(direction="max", weight=1.0)
        for k in get_target_columns(large_df)
    }


@pytest.fixture()
def gpytorch_model():
    return GaussianKernelRegression(["aa_1hot"], target_standardization=True, epochs=50)


# ======== Tests ==========


# ====== 1 ========
def test_artificial_fitness(large_df, small_df, gpytorch_model):
    fitness = af.PredictiveModelAF(
        large_df.rename_column("target a", "target summary variable"),
        gpytorch_model,
        "aa",
    )
    f = fitness(small_df)
    print(f["target summary variable"])


def test_policy_indepUCB():
    # pred_model = RFuniRep(True, "aa_1hot")
    pred_model = GaussianKernelRegression(["aa_1hot"])

    pol = IndepUCB(
        pred_model,
        exploration_factor=0.5,
        generator=gen.RandomGenerator(gen.ancestor.BestChoice(), "aa", 10, 2),
    )

    artif_wetlab = af.PredictiveModelAF(
        load_dataset("Company/CrHydA1_PE_REGR").rename_column(
            "target_activity_micromol_H2_per_microg_per_min",
            "target summary variable",
        )["whole_dataset"],
        GaussianKernelRegression(["aa_1hot"]),
        "aa",
    )

    measured = pred_model.add_encodings(artif_wetlab(artif_wetlab.starting_variants))

    # Now loop: get suggestions based on measurements, update model, get new suggestions
    for i in range(3):
        keys = jax.random.split(jax.random.PRNGKey(i), 2)
        pred_model.fit(keys[0], measured)
        suggestions = pol(
            keys[1], measured, 10
        )  # FIXME - this is taking for ever (or even is stuck) but it should not in the context of a test

        # Put the suggestions from our policy through the experiment
        new_measured = artif_wetlab(suggestions)

        # Add the new measurements to the old ones
        new_measured = new_measured.remove_columns(
            set(new_measured.features.keys())
            - {"aa_seq", "aa_1hot", "target summary variable"}
        )
        measured = concatenate_datasets([measured, new_measured])

        print(
            len(measured),
            measured.with_format("numpy")["target summary variable"].max(),
        )


def test_policy_batch_thompson(small_df, objectives):
    # pred_model = RFuniRep(True, "aa_1hot")
    pred_model = GaussianKernelRegression(["aa_1hot"])

    pol = BatchThompson(
        pred_model,
        generator=gen.CombinedGenerator(
            gen.RandomGenerator(gen.ancestor.BestChoice(), "aa", 10, 2),
            gen.DeepScanGenerator(gen.ancestor.BestChoice(), "aa"),
        ),
    )

    small_df = append_summary_variable(
        small_df,
        normalization=zscore,
        objectives=None,  # objectives -> None means all targets are maximized
        summary_variable_name="target summary variable",
    )

    artif_wetlab = af.PredictiveModelAF(
        small_df,
        GaussianKernelRegression(["aa_1hot"]),
        "aa",
        num_starting_variants=2,
    )
    artif_wetlab = af.CountAAcidAF(0)

    measured = pred_model.add_encodings(
        artif_wetlab(["MMMMMMMMMGGKLCGT", "MGGKLCGTMMMMMMMM"])
    )

    # Now loop: get suggestions based on measurements, update model, get new suggestions
    for i in range(5):
        keys = jax.random.split(jax.random.PRNGKey(i), 2)
        print(keys)
        pred_model.fit(keys[0], measured)
        suggestions = pol(keys[1], measured, 10)

        # Put the suggestions from our policy through the experiment
        new_measured = artif_wetlab(suggestions)

        # Add the new measurements to the old ones
        new_measured = new_measured.remove_columns(
            set(new_measured.features.keys())
            - {"aa_seq", "aa_1hot", "target summary variable"}
        )
        measured = concatenate_datasets([measured, new_measured])

        print(
            len(measured),
            measured.with_format("numpy")["target summary variable"].max(),
        )


# ====== 2 ========
def test_get_suggestions_gpytorch_DeepScanGenerator(
    large_df, total_suggestions, generator_ds, objectives, gpytorch_model
):
    total_no_suggestions = 32
    sugg = eng.get_suggestions(
        large_df,
        total_no_suggestions=total_no_suggestions,
        generator=generator_ds,
        objectives=objectives,
        pred_model=gpytorch_model,
    )

    assert np.isnan(np.array(sugg["decision criterion"])).sum() == 0

    assert (
        len(sugg) == total_no_suggestions
    ), f"Number of suggestions is not correct: {len(sugg)} instead of {total_no_suggestions}"


# FIXME (suggestions_gpytorch_FixedGenerator) ====== 3 ========
# def test_get_suggestions_gpytorch_FixedGenerator(
#     large_df,
#     total_suggestions,
#     generator_f,
#     objectives,
# ):
#     sugg = eng.get_suggestions(
#         large_df,
#         total_no_suggestions=32,
#         generator=generator_f,
#         objectives=objectives,
#     )

#     assert np.isnan(sugg["decision criterion"]).sum() == 0

#     assert len(sugg) == 32


if __name__ == "__main__":
    _, large_df, _ = get_small_large_df()

    total_suggestions = 10

    gpytorch_model = GaussianKernelRegression(
        ["aa_1hot"], target_standardization=True, epochs=50
    )

    wt = gen.ancestor.NamedChoice(large_df, "wildtype")(None, large_df)["aa_seq"][0]
    generator_ds = gen.DeepScanGenerator(wt, "aa", [1, 2, 3], True)

    objectives = {
        k: OptimizationObjective(direction="max", weight=1.0)
        for k in get_target_columns(large_df)
    }

    test_get_suggestions_gpytorch_DeepScanGenerator(
        large_df, total_suggestions, generator_ds, objectives, gpytorch_model
    )
