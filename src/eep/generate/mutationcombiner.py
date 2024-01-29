from typing import Any, Sequence
import datasets as ds
import numpy as onp
import pedata.mutation as pem
from pedata.constants import Mut
from .base import Generator
from .ancestor import AncestorChoice, FixedChoice
from ..util import PRNGKeyT


def add_new_pos(
    num_mutations_to_mutations: dict[int, list[list[Mut]]],
    mut_at_pos: dict[int, set[Mut]],
):
    """Add new positions to the mutation list.

    Args:
        num_mutations_to_mutations: A dictionary mapping the number of mutations to the actual mutations.
        mut_at_pos: The possible mutations at each position.
    """
    base_mutations = num_mutations_to_mutations[max(num_mutations_to_mutations.keys())]
    for mut_comb in base_mutations:
        # Find the positions that are already mutated
        mutated_pos = {mut.pos for mut in mut_comb}
        # Find the positions that are not yet mutated
        unmutated_pos = set(mut_at_pos.keys()).difference(mutated_pos)
        # Add a mutation at each unmutated position
        for pos in unmutated_pos:
            for mut in mut_at_pos[pos]:
                new_mut_comb = mut_comb + [mut]
                if len(new_mut_comb) not in num_mutations_to_mutations:
                    num_mutations_to_mutations[len(new_mut_comb)] = []
                num_mutations_to_mutations[len(new_mut_comb)].append(new_mut_comb)
    return num_mutations_to_mutations


class AllMutationsCombiner(Generator):
    """Combine mutations of measured sequences relative to a base sequence."""

    def __init__(
        self,
        base_seq_chooser: str | AncestorChoice,
        seq_type: str,
        allowed_no_of_mutations: Sequence[int] = None,
    ):
        """Constructor for AllMutationsCombiner.

        Args:
            base_seq_chooser: The sequence choose based on which to generate random mutations.
            seq_type: Sequence type of the base sequence (and candidates)
            allowed_no_of_mutations: Allowed total mutations. Defaults to None, in which case everything is allowed.
        """
        self.base_seq_chooser = FixedChoice.ensure_AncestorChoice(base_seq_chooser)
        self.seq_type = seq_type
        self.allowed_no_of_mutations = allowed_no_of_mutations

    def __call__(
        self,
        rngkey: PRNGKeyT,
        measured: ds.Dataset,
        exclude_seq: str | Sequence[str] | ds.Dataset,
    ) -> dict[str, Any] | ds.Dataset:
        """Generate sequences from combining mutations of measured sequences relative to a base sequence.

        Args:
            rngkey: The random number generator key.
            measured: The measured dataset.
            exclude_seq: The sequences to exclude from the generated dataset.

        Returns:
            The generated candidates.
        """
        measured_seq = measured[f"{self.seq_type}_seq"]

        base_seq = self.base_seq_chooser(rngkey, measured)
        assert len(base_seq) == 1, ValueError(
            "base_seq_chooser must return exactly one sequence"
        )
        base_seq = base_seq[f"{self.seq_type}_seq"][0]

        exclude_seq = Generator._concat_excluded(
            self.seq_type, exclude_seq, measured, base_seq
        )

        mutations = pem.extract.extract_mutation_namedtuples_from_sequences(
            measured_seq, base_seq
        )
        single_mutations = {item for row in mutations[0] for item in row}
        mut_at_pos = dict()
        for mut in single_mutations:
            mut_at_pos.setdefault(mut.pos, set()).add(mut)
        num_mutations_to_mutations = {1: [[mut] for mut in single_mutations]}
        for i in range(2, max(self.allowed_no_of_mutations) + 1):
            num_mutations_to_mutations = add_new_pos(
                num_mutations_to_mutations, mut_at_pos
            )
        # merge all mutations that are allowed
        mut_comb = []
        for no_of_mutations in self.allowed_no_of_mutations:
            mut_comb.extend(num_mutations_to_mutations[no_of_mutations])

        # mut_dict = pem.mutation_converter.namedtuple_to_dict_mut(mut_comb_filt[0])
        # mut_dataset = ds.Dataset.from_dict(mut_dict)
        self.seq_comb = onp.array(
            list(
                set(
                    pem.Mutation.apply_all_mutations(
                        mut_comb, parent=base_seq, offset=0
                    )
                )
            )
        )
        return Generator._included(self.seq_comb, exclude_seq)
