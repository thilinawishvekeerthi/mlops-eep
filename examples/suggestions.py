import jax.numpy as np
import datasets as ds
import pedata.config.paths as p
from pedata.util import get_target_columns, OptimizationObjective
import eep.generate as gen
import eep.engineer as eng


if __name__ == "__main__":
    total_suggestions = 32

    bc = gen.ancestor.BestChoice()

    # FIXME: JS: maybe add the dataset to a folder in the example directory and push it to the repo?
    df = ds.load_from_disk(p.PE_DATA_DIR / "CrHydA1_2022-9-6")

    # The following loads the dataset from a CSV file
    # df = dc.preprocess_data("example_dataset.csv")
    # configure to maximize all objectives (columns starting with "target")
    objectives = {
        k: OptimizationObjective(direction="max", weight=1.0)
        for k in get_target_columns(df)
    }

    wt = gen.ancestor.NamedChoice(df, set(("wildtype", "wildtype (WT)")))(None, df)[
        "aa_seq"
    ][0]

    gen_list = [
        gen.RandomGenerator(anc, "aa", 2, 2)
        for anc in (
            bc,
            # gen.ancestor.NamedChoice(df, "R227G_G425E"),
            gen.ancestor.FixedChoice(wt),
        )
    ]
    gen_list.extend(
        [
            gen.AllMutationsCombiner(gen.ancestor.NamedChoice(df, "wildtype"), "aa"),
            gen.ExhaustiveGenerator(bc, "aa", np.array([1, 3]) + 56),
            gen.DeepScanGenerator(bc, "aa", np.array([4, 5, 6]) + 56),
            gen.FixedGenerator("aa", [wt]),
        ]
    )
    generator = gen.CombinedGenerator(*gen_list)
    generator = gen.DeepScanGenerator(wt, "aa", np.array([1, 2, 3]) + 56, True)
    sugg = eng.get_suggestions(
        df,
        total_no_suggestions=total_suggestions,
        max_no_candidates=50,
        generator=generator,
        objectives=objectives,
    )
