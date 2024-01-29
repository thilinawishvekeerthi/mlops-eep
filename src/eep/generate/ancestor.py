from abc import ABC, abstractmethod
from typing import Callable, Sequence, Optional
import datasets as ds
from pedata.util import get_summary_variable
from ..util import PRNGKeyT


class AncestorChoice(Callable, ABC):
    """Abstract base class for ancestor choice strategies."""

    @abstractmethod
    def __call__(self, rngkey: PRNGKeyT, df: ds.Dataset) -> str | Sequence[str]:
        """Call ancestor choice algorithm represented by the object.

        Args:
            rngkey: Jax random number generator key.
            df : Huggingface dataset object including all sequences that where measured in the wet lab already.

        Returns:
            The ancestor(s) chosen.
        """


class NamedChoice(AncestorChoice, Callable):
    """Choose an ancestor by name."""

    def __init__(
        self,
        df: ds.Dataset,
        name: str | set[str] = set(("wildtype", "wildtype (WT)", "wt", "WT")),
    ) -> None:
        """Constructor for NamedChoice.

        Args:
            df : The dataset to choose from.
            name : The name of the ancestor to choose. Defaults to set(("wildtype", "wildtype (WT)")).
        """
        super().__init__()
        if isinstance(name, str):
            name = set((name,))
        self.name = name

    def __call__(self, rngkey: PRNGKeyT, df: ds.Dataset) -> Sequence[str]:
        """Call ancestor choice algorithm for named choice.

        Args:
            rngkey: Jax random number generator key.
            df: Huggingface dataset object including all sequences that where measured in the wet lab already.

        Raises:
            ValueError: If the ancestor name is not found in the dataset or if the ancestor name is not unique.

        Returns:
            Sequence[str]: The ancestor(s) chosen.

        Examples:
            >>> import datasets as ds
            >>> from eep.generate.ancestor import NamedChoice, FixedChoice, BestChoice
            >>> from pedata.util import (
            ...     OptimizationObjective,
            ...     append_summary_variable,
            ...     zscore,
            ... )
            >>> from pedata.mutation import Mutation
            >>> import jax.random as jr
            >>> # make a toy dataset
            >>> dataset_no_summary = ds.Dataset.from_dict(
            ...     {
            ...         "aa_mut": ["wt", "C9X_E11Y", "D10K"],
            ...         "aa_seq": ["ABCDEFGH", None, None],
            ...         "target a": [0, 1, 2],
            ...         "target b": [0, 1, 2],
            ...         "target c": [3, 1, 2],
            ...     }
            ... )
            >>> dataset_no_summary = dataset_no_summary.remove_columns("aa_seq").add_column(
            ...     "aa_seq", Mutation.apply_all_mutations(dataset_no_summary)
            ... )
            >>> # add a summary variable using the optimization objective
            >>> optimization_objective = {
            ...     "target a": OptimizationObjective(direction="max", weight=1.0),
            ...     "target b": OptimizationObjective(direction="min", weight=1.0),
            ...     "target c": OptimizationObjective(direction="fix", aim_for=2.0, weight=1.0),
            ... }
            >>> dataset_with_summary = append_summary_variable(
            ...     dataset=dataset_no_summary,
            ...     normalization=zscore,
            ...     objectives=optimization_objective,
            ...     summary_variable_name="target summary variable",
            ... )
            >>> from ..util import PRNGKeyT
            >>> from eep.generate.ancestor import BestChoice
            >>> from pedata.util import get_summary_variable, OptimizationObjective
            >>> # Example 4: NamedChoice
            >>> nc = NamedChoice(dataset_no_summary, "D10K")(jr.PRNGKey(0), dataset_no_summary)["aa_seq"]
            >>> print(nc)
            ['ABCKEFGH']
            >>> # Example 5: By default, get wildtype
            >>> nc = NamedChoice(dataset_no_summary)(jr.PRNGKey(0), dataset_no_summary)["aa_seq"]
            >>> print(nc)
            ['ABCDEFGH']
        """
        self.rval = df.filter(lambda x: x in self.name, input_columns="aa_mut")
        if len(self.rval) > 1:
            raise ValueError(f'Variant "{self.name}" found more than once.')
        elif len(self.rval) < 1:
            raise ValueError(f'Variant "{self.name}" not found.')
        return self.rval


class BestChoice(AncestorChoice, Callable):
    """Choose the ancestor with the best fitness according to target summary variable."""

    def __init__(
        self,
        ranks: list[int] = [0],
        dec_crit: str
        | dict[str, tuple[str, Optional[None]]] = "target summary variable",
    ) -> None:
        """Constructor for BestChoice.

        Args:
            ranks (list, optional): Ranks of ancestors to include. Defaults to [0], which will result in only returning the best ancestor.
        """
        super().__init__()
        self.dec_crit = dec_crit
        self.ranks = ranks

    def __call__(self, rngkey: PRNGKeyT, df: ds.Dataset) -> Sequence[str]:
        """Call ancestor choice algorithm represented by the object.

        Args:
            rngkey: Jax random number generator key.
            df: Huggingface dataset object including all sequences that where measured in the wet lab already.

        Returns:
            The ancestor(s) chosen.

        Examples:
        >>> import datasets as ds
        >>> from eep.generate.ancestor import NamedChoice, FixedChoice, BestChoice
        >>> from pedata.util import (
        ...     OptimizationObjective,
        ...     append_summary_variable,
        ...     zscore,
        ... )
        >>> from pedata.mutation import Mutation
        >>> import jax.random as jr
        >>> # make a toy dataset
        >>> dataset_no_summary = ds.Dataset.from_dict(
        ...     {
        ...         "aa_mut": ["wt", "C9X_E11Y", "D10K"],
        ...         "aa_seq": ["ABCDEFGH", None, None],
        ...         "target a": [0, 1, 2],
        ...         "target b": [0, 1, 2],
        ...         "target c": [3, 1, 2],
        ...     }
        ... )
        >>> dataset_no_summary = dataset_no_summary.remove_columns("aa_seq").add_column(
        ...     "aa_seq", Mutation.apply_all_mutations(dataset_no_summary)
        ... )
        >>> # add a summary variable using the optimization objective
        >>> optimization_objective = {
        ...     "target a": OptimizationObjective(direction="max", weight=1.0),
        ...     "target b": OptimizationObjective(direction="min", weight=1.0),
        ...     "target c": OptimizationObjective(direction="fix", aim_for=2.0, weight=1.0),
        ... }
        >>> dataset_with_summary = append_summary_variable(
        ...     dataset=dataset_no_summary,
        ...     normalization=zscore,
        ...     objectives=optimization_objective,
        ...     summary_variable_name="target summary variable",
        ... )
        >>> from ..util import PRNGKeyT
        >>> from eep.generate.ancestor import BestChoice
        >>> from pedata.util import get_summary_variable, OptimizationObjective

        >>> # Example 1, important for the webapp in the case of finding the base sequence to display:
        >>> anc = BestChoice(dec_crit=optimization_objective)(jr.PRNGKey(0), dataset_no_summary)
        >>> print(anc["aa_mut"], anc["aa_seq"]) #doctest: +NORMALIZE_WHITESPACE
        ['D10K'] ['ABCKEFGH']

        >>> # Example 2: Internal use in generators, not important for the webapp to find the base sequence to display:
        >>> anc = BestChoice(dec_crit="target summary variable")(
        ...     jr.PRNGKey(0), dataset_with_summary
        ... )
        >>> print(anc["aa_mut"], anc["aa_seq"])
        ['D10K'] ['ABCKEFGH']
        """
        # current_max = df[dec_crit_name][0]
        if isinstance(self.dec_crit, str):
            col = self.dec_crit
        else:
            col = "target summary variable (temporary)"
            summary = get_summary_variable(
                df.with_format("numpy"), objectives=self.dec_crit
            )
            df = df.add_column(col, summary)

        sorted = df.sort(col, reverse=True)
        if col == "target summary variable (temporary)":
            sorted = sorted.remove_columns(col)
        return sorted.select({i for i in self.ranks if i < len(df)})


class FixedChoice(AncestorChoice, Callable):
    """Choose a fixed sequence as ancestor. This ignores the dataset."""

    def __init__(self, fixed_sequence: str, seq_type: str = "aa") -> None:
        """Constructor for FixedChoice.

        Args:
            fixed_sequence: The sequence to use as ancestor.
            seq_type: The sequence type of `fixed_sequence`. Defaults to "aa".
        """
        self.rval = ds.Dataset.from_dict({f"{seq_type}_seq": [fixed_sequence]})

    def __call__(self, rngkey: PRNGKeyT, df: ds.Dataset) -> Sequence[str]:
        """Return the fixed sequence.

        Args:
            rngkey: Jax random number generator key.
            df Ignored parameter. Typically huggingface dataset object including all sequences that where measured in the wet lab already.

        Returns:
            The ancestor(s) chosen.

        Examples:
            >>> import datasets as ds
            >>> from eep.generate.ancestor import NamedChoice, FixedChoice, BestChoice
            >>> from pedata.util import (
            ...     OptimizationObjective,
            ...     append_summary_variable,
            ...     zscore,
            ... )
            >>> from pedata.mutation import Mutation
            >>> import jax.random as jr
            >>> # make a toy dataset
            >>> dataset_no_summary = ds.Dataset.from_dict(
            ...     {
            ...         "aa_mut": ["wt", "C9X_E11Y", "D10K"],
            ...         "aa_seq": ["ABCDEFGH", None, None],
            ...         "target a": [0, 1, 2],
            ...         "target b": [0, 1, 2],
            ...         "target c": [3, 1, 2],
            ...     }
            ... )
            >>> dataset_no_summary = dataset_no_summary.remove_columns("aa_seq").add_column(
            ...     "aa_seq", Mutation.apply_all_mutations(dataset_no_summary)
            ... )
            >>> # add a summary variable using the optimization objective
            >>> optimization_objective = {
            ...     "target a": OptimizationObjective(direction="max", weight=1.0),
            ...     "target b": OptimizationObjective(direction="min", weight=1.0),
            ...     "target c": OptimizationObjective(direction="fix", aim_for=2.0, weight=1.0),
            ... }
            >>> dataset_with_summary = append_summary_variable(
            ...     dataset=dataset_no_summary,
            ...     normalization=zscore,
            ...     objectives=optimization_objective,
            ...     summary_variable_name="target summary variable",
            ... )
            >>> from ..util  import PRNGKeyT
            >>> from eep.generate.ancestor import BestChoice
            >>> from pedata.util import get_summary_variable, OptimizationObjective
            >>> fc = FixedChoice("ABCDEFGH")(jr.PRNGKey(0), dataset_no_summary)["aa_seq"]
            >>> print(fc)
            ['ABCDEFGH']
        """
        return self.rval

    @classmethod
    def ensure_AncestorChoice(cls, arg: str | AncestorChoice) -> AncestorChoice:
        """Turn a string into a FixedChoice object. If the argument is already an AncestorChoice object, return it unchanged.

        Args:
            arg: The string to convert or AncestorChoice object.

        Returns:
            The string packed inside a FixedChoice object or unchanged AncestorChoice object.
        """
        if isinstance(arg, str):
            return cls(arg)
        else:
            return arg
