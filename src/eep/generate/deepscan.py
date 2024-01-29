from typing import Any, Dict, Iterable, Sequence
import datasets as ds
from Bio.Seq import Seq
import numpy as onp
import jax.numpy as np
from pedata.config.alphabets import valid_aa_alphabet
from .ancestor import AncestorChoice, FixedChoice
from .base import standard_bt, Generator, get_included_positions
from ..util import PRNGKeyT


def get_deep_scan(
    parent: str,
    parent_type: str,
    exclude_aa_seqs: Iterable[str] = set(),
    included_aa_positions: Iterable[int] = None,
) -> Dict[str, Any]:
    """Compute all mutations of single positions of `parent_seq`,

    Args:
        parent: The parent sequence to mutate.
        parent_type: Must be either "aa" or "dna", indicating the sequence type of `parent`.
        exclude_aa_seqs: Exclude these AA sequences from the result. Defaults to an empty set.
        included_aa_positions: Include only these AA positions in the sequence. Defaults to None, in which case all positions are included.

    Returns:
        Resulting mutated sequences, excluding the ones in `exclude_aa_seqs` or which translate to the ones on `exclude_aa_seqs` in case `parent_type == 'aa'`.
    """
    parent_type = parent_type.lower()

    if parent_type == "aa":
        aa_seq = parent
    else:
        aa_seq = str(Seq(parent).translate()).strip("*")
        dna_seq = parent

    rval = {}
    for t in ["aa"] if parent_type == "aa" else ["dna", "aa"]:
        for r in ["seq", "old", "mut_pos", "new", "mutations"]:
            rval[f"{t}_{r}"] = list()

    if included_aa_positions is None:
        included_aa_positions = range(len(aa_seq))
    for pos in included_aa_positions:
        aa_old_elem = aa_seq[pos]
        for aa_new_elem in valid_aa_alphabet:
            aa_new_seq = aa_seq[:pos] + aa_new_elem + aa_seq[pos + 1 :]

            if aa_old_elem == aa_new_elem or aa_new_seq in exclude_aa_seqs:
                continue

            aa_mut = aa_old_elem + str(pos) + aa_new_elem
            if parent_type == "aa":
                rval["aa_seq"].append(aa_new_seq)
                rval["aa_mut_pos"].append(pos)
                rval["aa_new"].append(aa_new_elem)
                rval["aa_old"].append(aa_old_elem)
                rval["aa_mutations"].append(aa_mut)
            else:  # "dna"
                dna_old_elem = parent[pos * 3 : (pos + 1) * 3]
                for dna_new_elem in standard_bt[aa_new_elem]:
                    dna_new_seq = (
                        dna_seq[: pos * 3] + dna_new_elem + dna_seq[pos * 3 + 3 :]
                    )
                    rval["aa_seq"].append(aa_new_seq)
                    rval["aa_mut_pos"].append(pos)
                    rval["aa_new"].append(aa_new_elem)
                    rval["aa_old"].append(aa_old_elem)
                    rval["aa_mutations"].append(aa_mut)
                    rval["dna_seq"].append(dna_new_seq)
                    rval["dna_mut_pos"].append(pos * 3)
                    rval["dna_new"].append(dna_new_elem)
                    rval["dna_old"].append(dna_old_elem)
                    rval["dna_mutations"].append((dna_new_elem, pos * 3, dna_old_elem))
    return rval


class DeepScanGenerator(Generator):
    """Candidate generator utilizing a digital deep mutational scan (DMS), i.e. all possible single mutations."""

    def __init__(
        self,
        base_seq_chooser: str | AncestorChoice,
        seq_type: str,
        aa_positions: list[int] = None,
        aa_positions_incl: bool = True,
        offset: int = 0,
    ):
        """Constructor for DeepScanGenerator.

        Args:
            base_seq_chooser: The sequence chooser resulting in the base sequence for which to apply the DMS. Alternatively a fixed sequence.
            seq_type: The type of `base_seq`, either 'aa' or 'dna'.
            aa_positions: AA sequence positions which to include (if `aa_positions_incl == True`; whitelist method) or exclude (if `aa_positions_incl == False`; blacklist method) from the DMS. Defaults to None, in which case all positions are included and `aa_positions_incl` is ignored. If provided, the implementation assumes that first position in sequence is numbered as "1" (unlike python, which assumes that positon to be "0").
            aa_positions_incl: Wether to use `aa_positions` as a white list or a black list. Only used if `aa_positions is not None`. Defaults to True.
            offset: Offset to shift `aa_positions` by. Defaults to 0.
        """
        self.base_seq_chooser = FixedChoice.ensure_AncestorChoice(base_seq_chooser)
        self.seq_type = seq_type
        self.offset = offset
        if aa_positions is None:
            self.aa_positions = None
        else:
            self.aa_positions = np.array(aa_positions)
        self.aa_positions_incl = aa_positions_incl
        self.aa_positions_incl = aa_positions_incl

    def __call__(
        self,
        rngkey: PRNGKeyT,
        measured: ds.Dataset,
        exclude_seq: str | Sequence[str] | ds.Dataset,
    ) -> dict[str, Any] | ds.Dataset:
        """Generate all variants for a deep mutational scan (DMS). I.e. all variants that result from all possible single mutations at all positions.

        Args:
            rngkey: Random number generator key.
            measured: Sequences for which measurement data was collected. To be avoided in resulting mutations.
            exclude_seq: Additional sequence(s) to be avoided in resulting mutations.

        Returns:
            The generated candidates.

        Raises:
            ValueError: If `base_seq_chooser` does not return exactly one sequence.
        """
        base_seq = self.base_seq_chooser(rngkey, measured)

        if len(base_seq) != 1:
            raise ValueError("base_seq_chooser must return exactly one sequence")

        base_seq = base_seq[f"{self.seq_type}_seq"][0]

        self.incl_pos = get_included_positions(
            len(base_seq), self.aa_positions, self.aa_positions_incl, self.offset
        )
        exclude_seq = Generator._concat_excluded(
            self.seq_type, exclude_seq, measured, base_seq
        )

        rval = get_deep_scan(
            base_seq,
            self.seq_type,
            exclude_seq,
            included_aa_positions=self.incl_pos,
        )[f"{self.seq_type}_seq"]

        return Generator._included(onp.array(rval), exclude_seq)
