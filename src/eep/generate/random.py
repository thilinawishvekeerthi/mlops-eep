from typing import Any, Callable, Sequence
import datasets as ds
import numpy as onp
import jax.numpy as np
from Bio.Seq import Seq
from jax.random import choice, permutation, randint, split
from ..util import Array, PRNGKeyT
from pedata.config.alphabets import valid_aa_alphabet
from .ancestor import AncestorChoice, FixedChoice
from .base import standard_bt, Generator, get_included_positions


def get_random_mutations(
    rngkey: PRNGKeyT,
    num_mutated_seq: int,
    parent_seq: Sequence[str],
    num_mutated_pos: int,
    func_pos_incl: Callable[[int], Sequence[int]],
    parent_type: str,
    parent_prob: Array = None,
) -> dict[str, Any]:
    """Generate `num_mutated_seq` random mutations of the sequences in `parent_seq`.

    Args:
        prng: Random seed for Jax random number generator
        num_mutated_seq: How many mutated sequences to return
        parent_seq: Sequences which will be mutated
        num_mutated_pos: Number of mutations in the amino acid sequence. If `parent_type == 'dna'`, this parameter will be in relation use the amino acid translation of the DNA parent sequence.
        func_pos_incl: Function which returns the positions in the parent sequence which can be mutated.
        parent_type: Must be either "aa" or "dna", depending on the type of sequences that `parent_seq` contains. If "dna", compute mutations on amino acid level and choose the DNA codon for the new amino acid uniformly at random from possible codons.
        parent_prob: Probability of chosing a parent from `parent_seq` (with replacement).
            (Default to None in which case parent is chosen uniformly at random)

    Returns:
        Dictionary containing mutated sequences and information on applied changes/mutations.
    """

    parent_type = parent_type.lower()
    splitted_prng = split(rngkey, num_mutated_seq + 1)
    parent_idcs = choice(
        splitted_prng[0], len(parent_seq), [num_mutated_seq], p=parent_prob
    ).squeeze()
    splitted_prng = splitted_prng[1:]

    rval = {}
    for t in ["aa"] if parent_type == "aa" else ["dna", "aa"]:
        for r in ["seq", "old", "mut_pos", "new", "mutations"]:
            rval[f"{t}_{r}"] = list()

    for i, parent_idx in enumerate(np.atleast_1d(parent_idcs).tolist()):
        aa_seq = parent = parent_seq[parent_idx]

        if parent_type == "dna":
            aa_seq = str(Seq(parent).translate()).strip("*")
            dna_seq = parent
        num_mutated_pos = min(num_mutated_pos, len(aa_seq))
        # lists for storing changed position and values
        prng_perm, prng_mutation, prng_codon = split(splitted_prng[i], 3)
        lst_aa_new = []
        lst_aa_old = []
        lst_aa_mutations = []
        if parent_type == "dna":
            lst_dna_mutations = []
            lst_dna_new = []
            lst_dna_old = []
        aa_pos_target = np.array(
            [
                permutation(
                    prng_perm, np.array(list(func_pos_incl(len(aa_seq))), dtype=int)
                )[:num_mutated_pos],
                randint(
                    prng_mutation, [num_mutated_pos], 0, len(valid_aa_alphabet) - 1
                ).reshape(-1),
            ]
        )
        for pos, targ in aa_pos_target.T:
            aa_old_elem = aa_seq[pos]

            # minimize alphabet such that aa_old_elem cannot be drawn
            tmp_alphabet = valid_aa_alphabet.copy()
            try:
                tmp_alphabet.remove(aa_old_elem)
            except ValueError as ve:
                print(f"Couldn't remove {aa_old_elem} from alphabet")
                raise ve

            aa_new_elem = tmp_alphabet[targ]

            lst_aa_new.append(aa_new_elem)
            lst_aa_old.append(aa_old_elem)
            lst_aa_mutations.append(aa_old_elem + str(pos) + aa_new_elem)
            if parent_type == "aa":
                aa_seq = aa_seq[:pos] + aa_new_elem + aa_seq[pos + 1 :]
            else:
                dna_old_elem = parent[pos * 3 : (pos + 1) * 3]
                codons = standard_bt[aa_new_elem]
                codon_idx = randint(prng_mutation, [1], 0, len(codons) - 1).squeeze()
                dna_new_elem = codons[codon_idx]

                lst_dna_old.append(dna_old_elem)
                lst_dna_new.append(dna_new_elem)
                lst_dna_mutations.append((dna_old_elem, pos * 3, dna_new_elem))
                dna_seq = dna_seq[: pos * 3] + dna_new_elem + dna_seq[pos * 3 + 3 :]
        if parent_type == "dna":
            aa_seq = str(Seq(dna_seq).translate().strip("*"))

        rval["aa_seq"].append(aa_seq)
        rval["aa_mut_pos"].append(aa_pos_target[0, :])
        rval["aa_new"].append(lst_aa_new)
        rval["aa_old"].append(lst_aa_old)
        rval["aa_mutations"].append(lst_aa_mutations)
        if parent_type == "dna":
            rval["dna_seq"].append(dna_seq)
            rval["dna_mut_pos"].append(aa_pos_target[0, :] * 3)
            rval["dna_new"].append(lst_dna_new)
            rval["dna_old"].append(lst_dna_old)
            rval["dna_mutations"].append(lst_dna_mutations)
    return rval


class RandomGenerator(Generator):
    """Random mutation candidate generator, a digital version of error prone PCR."""

    def __init__(
        self,
        base_seq_chooser: str | AncestorChoice,
        seq_type: str,
        num_mutated_seq: int,
        num_mutated_pos: int,
        aa_positions: list[int] = None,
        aa_positions_incl: bool = True,
        offset: int = 0,
    ) -> None:
        """Constructor for RandomGenerator.

        Args:
            base_seq_chooser : The sequence choose based on which to generate random mutations.
            seq_type: The type of `base_seq`, either 'aa' or 'dna'.
            num_mutated_seq: The total number of mutated sequences to generate.
            num_mutated_pos: The number of mutated positions per generated sequence.
            aa_positions: AA sequence positions which to include (if `aa_positions_incl == True`; whitelist method) or exclude (if `aa_positions_incl == False`; blacklist method) from the DMS. Defaults to None, in which case all positions are included and `aa_positions_incl` is ignored. If provided, the implementation assumes that first position in sequence is numbered as "1" (unlike python, which assumes that positon to be "0").
            aa_positions_incl: Wether to use `aa_positions` as a white list or a black list. Only used if `aa_positions is not None`. Defaults to True.
            offset: Offset to shift `aa_positions` by. Defaults to 0.
        """
        super().__init__()
        self.base_seq_chooser = FixedChoice.ensure_AncestorChoice(base_seq_chooser)
        self.num_mutated_seq = num_mutated_seq
        self.num_mutated_pos = num_mutated_pos
        self.seq_type = seq_type
        self.offset = offset
        if aa_positions is None:
            self.aa_positions = None
        else:
            self.aa_positions = np.array(aa_positions)
        self.aa_positions_incl = aa_positions_incl

    def __call__(
        self,
        rngkey: PRNGKeyT,
        measured: ds.Dataset,
        exclude_seq: str | Sequence[str] | ds.Dataset,
    ) -> dict[str, Any] | ds.Dataset:
        """Call random mutation generator candidate generator.

        Args:
            rngkey: Random number generator key.
            measured: Sequences for which measurement data was collected. To be avoided in resulting mutations.
            exclude_seq: Additional sequence(s) to be avoided in resulting mutations.

        Returns:
            The generated mutated sequences.
        """
        rng1, rng2 = split(rngkey)  # FIXME replace with split_key from eep?
        measured = measured.with_format(None)
        base_seq = self.base_seq_chooser(rng1, measured)[f"{self.seq_type}_seq"]
        exclude_seq = Generator._concat_excluded(
            self.seq_type, exclude_seq, measured, base_seq
        )
        mut = get_random_mutations(
            rng2,
            self.num_mutated_seq,
            base_seq,
            self.num_mutated_pos,
            lambda x: get_included_positions(
                x, self.aa_positions, self.aa_positions_incl, self.offset
            ),
            self.seq_type,
            None,
        )
        # mut = pd.DataFrame(mut)
        # inc = Mutator._included(mut[f"{self.parent_type}_seq"].values, parent_seq, exclude_seq)
        rval = Generator._included(onp.array(mut[f"{self.seq_type}_seq"]), exclude_seq)
        if len(rval) == 0:
            raise ValueError("No mutations generated from random generator")
        return rval
