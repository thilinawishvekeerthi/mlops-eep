from typing import Sequence
import eep.generate as gen
import pedata.mutation as pem
import datasets as ds
import jax.random as rand
import numpy as onp


def test_exhaustive():
    base_seq = "MAGLYITR"
    small_df = ds.Dataset.from_dict({"aa_seq": [base_seq]})
    large_df = ds.Dataset.from_dict({"aa_seq": ["MYGLYITR", base_seq]})
    gen1 = gen.ExhaustiveGenerator(base_seq, "aa", [2, 5])
    gen2 = gen.ExhaustiveGenerator(
        base_seq,
        "aa",
        [
            2,
        ],
    )
    v1 = gen1(rand.PRNGKey(1), large_df, [base_seq])
    assert v1.size == 398
    assert (
        len(
            set(list(v1)).intersection(
                [
                    "MYGLYITR",
                    "MAGLYITL",
                ]
            )
        )
        == 0
    )
    v2 = gen2(rand.PRNGKey(1), small_df, ["MYGLYITR", base_seq])
    assert v2.size == 18
    assert (
        len(
            set(list(v2)).intersection(
                [
                    "MKGLYITR",
                    "CMMLATG",
                ]
            )
        )
        == 1
    )


def test_fixed():
    resulting_sequences = ["MYGLYITR", "MAGLYITL"]
    gen1 = gen.FixedGenerator("aa", resulting_sequences)

    v1 = gen1(rand.PRNGKey(1), "DOESNTMATTER", [])
    assert v1.size == 2
    assert len(set(list(v1)).intersection(set(resulting_sequences))) == 2


def test_fixed_from_mutcode():
    base_seq = "MAGLYITR"
    gen1 = gen.FixedGenerator.from_mutcode("aa", ["A20Y", "G21L"], base_seq, offset=-18)
    v1 = gen1(rand.PRNGKey(1), "DOESNTMATTER", [])
    assert v1.size == 2
    assert (
        len(
            set(list(v1)).intersection(
                {
                    "MYGLYITR",
                    "MALLYITR",
                }
            )
        )
        == 2
    )


def test_dms():
    base_seq = "MAGLYITR"
    measured = ds.Dataset.from_dict({"aa_seq": [base_seq]})
    gen1 = gen.DeepScanGenerator(base_seq, "aa", [2, 5])
    gen2 = gen.DeepScanGenerator(
        base_seq,
        "aa",
        [
            2,
        ],
    )
    v1 = gen1(
        rand.PRNGKey(1),
        measured,
        [
            "MYGLYITR",
        ],
    )
    assert v1.size == 2 * 19 - 1
    assert (
        len(
            set(list(v1)).intersection(
                [
                    "MYGLYITR",
                    "MAGLYITL",
                ]
            )
        )
        == 0
    )
    v2 = gen2(rand.PRNGKey(1), measured, ["MYGLYITR", "MAGLYITR"])
    assert v2.size == 18
    assert (
        len(
            set(list(v2)).intersection(
                [
                    "MKGLYITR",
                    "CMMLATG",
                ]
            )
        )
        == 1
    )
    gen3 = gen.DeepScanGenerator(
        base_seq,
        "aa",
        [
            2,
        ],
        aa_positions_incl=False,
    )
    v3 = gen3(rand.PRNGKey(1), measured, [])
    assert v3.size == 19 * (len(base_seq) - 1)
    gen4 = gen.DeepScanGenerator(
        base_seq,
        "aa",
    )
    v4 = gen4(rand.PRNGKey(1), measured, [])
    assert v4.size == 19 * len(base_seq)


def test_random():
    base_seq = "MAGLYITR"
    measured_df = ds.Dataset.from_dict({"aa_seq": base_seq})
    base_arr = onp.array(list(base_seq))
    gen1 = gen.RandomGenerator(base_seq, "aa", 4, 2)
    gen2 = gen.RandomGenerator(base_seq, "aa", 2, 3)

    v1 = gen1(
        rand.PRNGKey(1),
        measured_df,
        ds.Dataset.from_dict(
            {
                "aa_seq": [
                    "MYGLYITR",
                ]
            }
        ),
    )
    assert v1.size == 4
    for s in v1:
        assert onp.sum(onp.array(list(s)) != base_arr) == 2
    v2 = gen2(
        rand.PRNGKey(1),
        measured_df,
        ds.Dataset.from_dict({"aa_seq": ["MYGLYITR", base_seq]}),
    )
    assert v2.size == 2
    for s in v2:
        assert onp.sum(onp.array(list(s)) != base_arr) == 3


def test_comb():
    import pedata.mutation as pem

    base_seq = "MAGLYITR"
    five_variants = [
        "CAGLYITR",
        "MCGLYITR",
        "MACLYITR",
        "MAGCYITR",
        "MAGLCITR",
    ]
    df = ds.Dataset.from_dict({"aa_seq": five_variants})

    gen1 = gen.AllMutationsCombiner(base_seq, "aa", [2])
    v1 = gen1(
        rand.PRNGKey(1),
        df,
        [
            base_seq,
        ],
    )
    # combinations of 2 mutations, no duplicates
    assert v1.size == 10

    gen2 = gen.AllMutationsCombiner(base_seq, "aa", [2, 3])
    v2 = gen2(
        rand.PRNGKey(1),
        df,
        [
            base_seq,
        ],
    )
    assert v2.size == 20

    gen3 = gen.AllMutationsCombiner(base_seq, "aa", [2])
    v3 = gen3(
        rand.PRNGKey(1),
        ds.Dataset.from_dict(
            {
                "aa_seq": five_variants
                + [
                    "AAGLYITR",
                ]
            }
        ),
        [
            base_seq,
        ],
    )
    # combinations of 2 mutations, no duplicates
    assert v3.size == 14

    assert set(v1).issubset(set(v3))
    assert set(v1).issubset(set(v2))


def test_get_included_positions():
    assert list(gen.get_included_positions(5, onp.array([2, 5]), True, 0)) == [1, 4]
    assert list(gen.get_included_positions(5, onp.array([2, 5]), True, -1)) == [0, 3]
    assert list(gen.get_included_positions(5, onp.array([2, 5]), False, -1)) == [
        1,
        2,
        4,
    ]
    assert list(gen.get_included_positions(5, onp.array([2, 5]), False, 0)) == [0, 2, 3]
    assert list(gen.get_included_positions(5, None, False, 0)) == [0, 1, 2, 3, 4]

    try:
        gen.get_included_positions(8, onp.array([1, 3]), True, -3)
    except ValueError:
        pass
    else:
        assert False, "Invalid offset Should have raised ValueError"
    # assert not any(onp.array(list(gen.get_included_positions(8, onp.array([1, 3]), True, -3))) < 0)


def extract_mutation_positions(mut: Sequence[str]) -> set[int]:
    """Extract mutation positions from a mutations code gives as strings

    Args:
        mut (Sequence[str]): Mutation codes

    Returns:
        set[int]: Extracted mutation positions
    """
    # actual mutation position resulting from splitting each mutation by character "_" and
    # extract position number only for each single mutation (get rid of amino acid letters)

    # m_pos = [m['position'] for m in mut]
    # # Flatten m_pos to get rid of nested lists
    # return {item for sublist in m_pos for item in sublist}

    positions = []

    for mutation in mut:
        if "_" in mutation:
            split_mutations = mutation.split("_")
            for m in split_mutations:
                positions.append(int(m[1]))

        else:
            positions.append(int(mutation[1]))

    return positions


def test_offset_and_incl_positions():
    base_seq = "MAGLYITR"
    positions = onp.array([6, 7])
    offset = 0
    gen_not_incl = gen.CombinedGenerator(
        gen.DeepScanGenerator(base_seq, "aa", positions, aa_positions_incl=False),
        gen.RandomGenerator(base_seq, "aa", 40, 2, positions, aa_positions_incl=False),
        offset=offset,
    )
    gen_incl = gen.CombinedGenerator(
        gen.DeepScanGenerator(base_seq, "aa", positions, aa_positions_incl=True),
        gen.RandomGenerator(base_seq, "aa", 40, 2, positions, aa_positions_incl=True),
        offset=offset,
    )

    def mut_not_incl(off):
        tmp_off = gen_not_incl.offset
        gen_not_incl.offset = off
        rval = pem.extract.extract_mutation_str_from_sequences(
            gen_not_incl(
                rand.PRNGKey(1), ds.Dataset.from_dict({"aa_seq": [base_seq]}), []
            ),
            base_seq,
            off,
        )
        gen_not_incl.offset = tmp_off
        return set(extract_mutation_positions(rval))

    def mut_incl(off):
        tmp_off = gen_incl.offset
        gen_incl.offset = off
        rval = pem.extract.extract_mutation_str_from_sequences(
            gen_incl(rand.PRNGKey(1), ds.Dataset.from_dict({"aa_seq": [base_seq]}), []),
            base_seq,
            off,
        )
        gen_incl.offset = tmp_off
        return set(extract_mutation_positions(rval))

    # check that forbidden positions are not included in actual mutation positions
    pos_not_incl = mut_not_incl(offset)
    assert not any([x in positions for x in pos_not_incl])
    # check that only allowed positions are included in actual mutation positions
    pos_incl = mut_incl(offset)
    assert all([x in positions for x in pos_incl])

    # repeat, but now setting different offsets
    for offset in range(-5, -1):
        # check that forbidden positions are not included in actual mutation positions
        pos_not_incl = mut_not_incl(offset)
        assert not any([x in positions for x in pos_not_incl])
        # check that only allowed positions are included in actual mutation positions
        pos_incl = mut_incl(offset)
        assert all([x in positions for x in pos_incl])
