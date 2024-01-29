from typing import Any, Sequence
import jax.numpy as np
import numpy as onp
from ..util import PRNGKeyT
import datasets as ds
from pedata.config.alphabets import valid_aa_alphabet, valid_dna_alphabet
import pedata.mutation as pedm
from .ancestor import AncestorChoice, FixedChoice
from .base import Generator


class ExhaustiveGenerator(Generator):
    """Combine all possible mutations for positions passed."""

    def __init__(
        self,
        base_seq_chooser: str | AncestorChoice,
        seq_type: str,
        natural_positions: Sequence[int],
        offset: int = 0,
    ) -> None:
        """Constructor for exhaustive mutation generator.

        Args:
            base_seq_chooser: Base for the exhaustive generator
            seq_type: The type of `base_seq`, either 'aa' or 'dna'.
            natural_positions: The positions in the parent sequences to mutate. Assumes that first position in sequence is numbered as "1" (unlike python, which assumes that positon to be "0").
            offset: Offset to shift `positions` by. Defaults to 0.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        assert natural_positions is not None, ValueError(
            "Parameter `natural_positions` must not be None."
        )
        assert len(natural_positions) <= 3 and len(natural_positions) > 0, ValueError(
            "Parameter `natural_positions` must have length between 1 and 3."
        )
        self.offset = offset
        self.base_seq_chooser = FixedChoice.ensure_AncestorChoice(base_seq_chooser)
        self.natural_positions = np.array(natural_positions)

        self.seq_type = seq_type.lower()
        if self.seq_type == "aa":
            self.alphabet = valid_aa_alphabet
        elif self.seq_type == "dna":
            self.alphabet = valid_dna_alphabet
        else:
            raise ValueError(f"Parent type '{seq_type}' not recognized")

    def __call__(
        self,
        rngkey: PRNGKeyT,
        measured: ds.Dataset,
        exclude_seq: str | Sequence[str] | ds.Dataset,
    ) -> dict[str, Any] | ds.Dataset:
        """Generate exhaustive mutations.

        Args:
            rngkey: The random number generator key.
            measured: The measured dataset.
            exclude_seq: The sequences to exclude from the generated dataset.

        Returns:
            The generated candidates.
        """
        base_seq = self.base_seq_chooser(rngkey, measured)
        exclude_seq = Generator._concat_excluded(
            self.seq_type, exclude_seq, measured, base_seq
        )
        assert len(base_seq) == 1, ValueError(
            "base_seq_chooser must return exactly one sequence"
        )
        base_seq = base_seq[f"{self.seq_type}_seq"][0]
        # TODO: check if the following can be implemented in a cleaner way
        seq = list()
        mut_parent = []
        for pos in list(np.array(self.natural_positions) - 1 + self.offset):
            mut_parent.append(list())
            # FIXME: use Mut(Position, Source, Target) instead of dicts
            for t in self.alphabet:
                if t != base_seq[pos]:
                    mut_parent[-1].append(
                        {
                            "source": [base_seq[pos]],
                            "target": [t],
                            "position": np.array([pos]),
                        }
                    )

            if len(mut_parent) == 2:
                namedtuple_mutations = []
                for i in range(2):
                    namedtuple_mutations.append(
                        pedm.mutation_converter.dict_to_namedtuple_mut(mut_parent[i])
                    )
                mut_parent = namedtuple_mutations
                tmp = pedm.Mutation.generate_variant_mutation_combinations(
                    *mut_parent
                )  # pedm.combine_mutations_of_variants(*mut_parent)
                tmp.extend(mut_parent[0])
                tmp.extend(mut_parent[1])
                mut_parent = [tmp]

            else:
                for i in range(len(mut_parent)):
                    mut_parent[i] = pedm.mutation_converter.dict_to_namedtuple_mut(
                        mut_parent[i]
                    )

        for m in mut_parent[0]:
            if not isinstance(m, list):
                m = [m]
            # seq.append(pedm.mutation_converter.convert_variant_mutation_to_str(m))
            seq.append(pedm.Mutation.apply_variant_mutations(m, base_seq))
            # mut.append(pedm.mutation_converter.convert_variant_mutation_to_str(m))

        return Generator._included(onp.array(seq), exclude_seq)
