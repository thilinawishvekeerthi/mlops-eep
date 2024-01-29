from abc import ABC
from typing import Sequence, Union, Any
import datasets as ds
import jax.random as jr
import numpy as onp
from Bio.Data.CodonTable import standard_dna_table
from ..util import PRNGKeyT, Array


def make_ambiguous_back_table(forward_table):
    """Back a back-table for BioPython CodonTable.
    Args:
        forward_table: Forward table to make ambiguous back-table for.
    Returns:
        dict: Ambiguous back-table.
    """
    bt = {}
    for codon in forward_table:
        if forward_table[codon] in bt:
            bt[forward_table[codon]].append(codon)
        else:
            bt[forward_table[codon]] = [codon]
    return bt


standard_bt = make_ambiguous_back_table(standard_dna_table.forward_table)


def get_included_positions(
    base_seq_len: int, positions: Array, positions_incl: bool, offset: int
) -> Sequence[int]:
    """Get the positions to include in a mutation procedure.

    Args:
        base_seq_len: Length of the sequence mutations are based upon.
        positions: Positions to include or exclude. If None, all positions are included.
        positions_incl: Wethter to include or exclude the positions in `positions`. Ignored if `positions` is None.
        offset: The offset to add to the positions in `positions`.

    Returns:
        Sequence[int]: The positions to include in a mutation procedure.
    """

    if positions is None:
        incl_pos = range(base_seq_len)
    else:
        positions = positions - 1 + offset
        if any(positions < 0) or any(positions >= base_seq_len):
            raise ValueError(
                f"Invalid offset: {offset} (base_seq_len {base_seq_len}, minimum pos {min(positions) + 1 - offset}, maximum pos {max(positions) + 1 - offset})"
            )
        positions = positions.tolist()
        if positions_incl:
            incl_pos = positions
        else:
            incl_pos = set(range(base_seq_len)).difference(positions)
    return incl_pos


class Generator(ABC):
    """Mutator baseclass. Mutators are callable objects that mutate parent sequences based on some algorithm."""

    @staticmethod
    def _concat_excluded(
        seq_type: str, *seq_src: Union[str, Sequence[str], ds.Dataset]
    ) -> list[str]:
        """Concatenate sequences to be excluded from suggestions.

        Args:
            seq_type: The type of `base_seq`, either 'aa' or 'dna'.
            *seq_src: The sequences to be processed.

        Returns:
            list[str]: The concatenated sequences.
        """
        exclude_seq = set()
        for s in seq_src:
            if isinstance(s, str):
                exclude_seq.add(s)
            else:
                if isinstance(s, ds.Dataset):
                    s = s.with_format(None)[f"{seq_type}_seq"]
                exclude_seq = exclude_seq.union(s)

        return list(exclude_seq)

    @staticmethod
    def _included(
        generation_result: Sequence[str],
        exclude_seq: Sequence[str],
        return_bool=False,
    ) -> Array | Sequence:
        """Filter mutation results, removing parent sequences and excluded sequences.

        Args:
            generation_result: The generated sequences as strings.
            exclude_seq: Sequences to be excluded from the returned list.
            return_bool: If true, return an indicator sequence instead of the included sequences. Defaults to False.

        Returns:
            Boolean indicator or included sequences.
        """
        excl = onp.array(list(set(exclude_seq)))
        is_incl = onp.isin(generation_result, excl, invert=True)
        if return_bool:
            return is_incl
        else:
            return generation_result[is_incl]

    def __call__(
        self,
        rngkey: PRNGKeyT,
        measured: ds.Dataset,
        exclude_seq: str | Sequence[str] | ds.Dataset,
    ) -> dict[str, Any] | ds.Dataset:
        """The call signature for Mutators.

        Args:
            rngkey: Random number generator key.
            measured: Sequences for which measurement data was collected. To be avoided in resulting mutations.
            exclude_seq: Additional sequence(s) to be avoided in resulting mutations.

        Returns:
            The generated mutated sequences.
        """
        pass


class CombinedGenerator(Generator):
    """Combine a couple of generator objects into a single object that executes them sequentially and collects the results."""

    def __init__(self, *generators: Generator, offset: int = 0) -> None:
        """Constructor for CombinedGenerator.

        Args:
            *generators (Generator): The generators to be combined.
            offset (int, optional): The offset to add to the positions in `positions`. Defaults to 0.

        Example:
            c = CombinedGenerator(MutationCombiner(base_sequence, seq_type), DeepScanGenerator(base_sequence, seq_type))
            sequence_candidates = c(rngkey, huggingface_dataset_object, excluded_seqs)
        """
        super().__init__()
        self.gen = generators
        self.offset = offset

    @property
    def offset(self) -> int:
        """Offest to add to user-defined positions

        Returns:
            int: The offset. Assumed to be 0 if not set.
        """
        return self._offset

    @offset.setter
    def offset(self, value: int):
        """Set the offset to add to user-defined positions.

        Args:
            value (int): The offset.
        """
        self._offset = value
        for i, m in enumerate(self.gen):
            m.offset = value

    def __call__(
        self,
        rngkey: PRNGKeyT,
        measured: ds.Dataset,
        exclude_seq: str | Sequence[str] | ds.Dataset,
    ) -> dict[str, Any] | ds.Dataset:
        """Call all generators in sequence and collect the results.

        Args:
            rngkey: Random number generator key.
            measured: Sequences for which measurement data was collected. To be avoided in resulting mutations.
            exclude_seq: Additional sequence(s) to be avoided in resulting mutations.

        Returns:
            The generated mutated sequences.
        """
        rval = set()
        rng = jr.split(rngkey, len(self.gen))
        for i, m in enumerate(self.gen):
            rval.update(list(m(rng[i], measured, exclude_seq)))
        rval = onp.array(list(rval))
        return rval
