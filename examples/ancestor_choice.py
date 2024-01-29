import datasets as ds
from eep.generate.ancestor import NamedChoice, FixedChoice, BestChoice
from pedata.util import (
    OptimizationObjective,
    append_summary_variable,
    zscore,
)
from pedata.mutation import Mutation
import jax.random as jr

dataset_no_summary = ds.Dataset.from_dict(
    {
        "aa_mut": ["wt", "C9X_E11Y", "D10K"],
        "aa_seq": ["ABCDEFGH", None, None],
        "target a": [0, 1, 2],
        "target b": [0, 1, 2],
        "target c": [3, 1, 2],
    }
)

dataset_no_summary = dataset_no_summary.remove_columns("aa_seq").add_column(
    "aa_seq", Mutation.apply_all_mutations(dataset_no_summary)
)

optimization_objective = {
    "target a": OptimizationObjective(direction="max", weight=1.0),
    "target b": OptimizationObjective(direction="min", weight=1.0),
    "target c": OptimizationObjective(direction="fix", aim_for=2.0, weight=1.0),
}

dataset_with_summary = append_summary_variable(
    dataset=dataset_no_summary,
    normalization=zscore,
    objectives=optimization_objective,
    summary_variable_name="target summary variable",
)

# Example 1, important for the webapp in the case of finding the base sequence to display:
anc = BestChoice(dec_crit=optimization_objective)(jr.PRNGKey(0), dataset_no_summary)
# print(anc["aa_mut"], anc["aa_seq"])
assert anc["aa_mut"] == ["D10K"] and anc["aa_seq"] == ["ABCKEFGH"], "Example 1 failed"


# Example 2: Internal use in generators, not important for the webapp to find the base sequence to display:
anc = BestChoice(dec_crit="target summary variable")(
    jr.PRNGKey(0), dataset_with_summary
)
# print(anc["aa_mut"], anc["aa_seq"])
assert anc["aa_mut"] == ["D10K"] and anc["aa_seq"] == ["ABCKEFGH"], "Example 2 failed"

# Example 3: FixedChoice
fc = FixedChoice("ABCDEFGH")(jr.PRNGKey(0), dataset_no_summary)["aa_seq"]
# print(fc)
assert fc == ["ABCDEFGH"], "Example 3 failed"

# Example 4: NamedChoice
nc = NamedChoice(dataset_no_summary, "D10K")(jr.PRNGKey(0), dataset_no_summary)[
    "aa_seq"
]
# print(nc)
assert fc == ["ABCDEFGH"], "Example 4 failed"

# Example 5: By default, get wildtype
nc = NamedChoice(dataset_no_summary)(jr.PRNGKey(0), dataset_no_summary)["aa_seq"]
# print(nc)
assert fc == ["ABCDEFGH"], "Example 5 failed"
