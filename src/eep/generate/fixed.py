from typing import Any, Iterable, Sequence
import datasets as ds
import numpy as onp
import pedata as ped
from .base import Generator
from ..util import PRNGKeyT


class FixedGenerator(Generator):
    """Generator that returns a predefined, fixed list of sequences. This can be used for incorporating researchers prior knowledge in sequence design."""

    def __init__(self, seq_type: str, resulting_sequences: Iterable[str]):
        """Constructor for FixedGenerator.

        Args:
            seq_type (str): The type of `base_seq`, either 'aa' or 'dna'.
            resulting_sequences (Iterable[str]): An iterable containing all sequences that this generator will return.
        """
        self.seq_type = seq_type
        self.resulting_sequences = list(set(resulting_sequences))

    @classmethod
    def from_mutcode(
        cls, seq_type: str, mutcodes: Iterable[str], parent_seq: str, offset: int = None
    ) -> "FixedGenerator":
        """Create a FixedGenerator from mutation code strings.

        Args:
            seq_type: The type of `base_seq`, either 'aa' or 'dna'.
            mutcodes: An iterable of mutcode strings.
            parent_seq: The parent sequence relative to which the mutcodes are defined.
            offset: The offset to apply to the mutcodes.

        Returns:
            FixedGenerator: The FixedGenerator object.
        """
        mutcodes = list(set(mutcodes))
        mutations = [
            ped.mutation.Mutation.parse_variant_mutations(code) for code in mutcodes
        ]
        if offset is None:
            offset = ped.mutation.Mutation.estimate_offset(
                [element for m in mutations for element in m],
                parent_seq,
                most_likely=True,
            )
        resulting_sequences = ped.mutation.Mutation.apply_all_mutations(
            mutations,
            parent=parent_seq,
            offset=offset,
            check_validity=True,
        )

        return cls(seq_type, resulting_sequences)

    def __call__(
        self,
        rngkey: PRNGKeyT,
        measured: ds.Dataset,
        exclude_seq: str | Sequence[str] | ds.Dataset,
    ) -> dict[str, Any] | ds.Dataset:
        """'Generate' candidates, i.e. return the fixed list of sequences.

        Args:
            rngkey: The random number generator key.
            measured: The measured dataset.
            exclude_seq: The sequences to exclude from the generated dataset.

        Returns:
            The generated candidates.
        """
        if isinstance(exclude_seq, ds.Dataset):
            exclude_seq = exclude_seq[f"{self.seq_type}_seq"]
        elif isinstance(exclude_seq, str):
            exclude_seq = [exclude_seq]

        return Generator._included(onp.array(self.resulting_sequences), exclude_seq)
