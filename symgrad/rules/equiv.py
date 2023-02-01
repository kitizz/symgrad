# Copyright, Christopher Ham, 2022

from collections.abc import Sequence
import logging
from typing import Any

from ..constant import Constant
from ..exact import exact
from ..expression import Expression
from .knowledgebase import Knowledgebase, the_knowledgebase

__all__ = [
    "equiv",
    "equiv_seq",
    "is_equivalent",
]


class Equivalent:
    expr: Expression

    def __init__(self, expr: Expression):
        self.expr = expr

    def __eq__(self, other):
        """Equality testing in the reverse order relies on Equivalent being a different
        type from any other Generator. And other Generators using _hash() as an initial check.
        """
        if isinstance(other, Expression):
            return is_equivalent(self.expr, other)

        if isinstance(other, Equivalent):
            return is_equivalent(self.expr, other.expr)

        if isinstance(self.expr, Constant):
            return self.expr.value == other

        return NotImplemented

    def __str__(self):
        return f"Equivalent({str(self.expr)})"

    def __repr__(self):
        return f"Equivalent({repr(self.expr)})"


def equiv_seq(expr_list: Sequence[Any]) -> list[Equivalent]:
    return [equiv(v) for v in expr_list]


def equiv(expr) -> Equivalent:
    """Ensures comparisons with other Generators compare that the expression matches exactly.
    As opposed to merely being equivalent.

    Examples:
        equiv(x + 1) == (x + 1) -> True
        equiv(x + 1) == (1 + x) -> True
        (x + 1) == equiv(1 + x) -> True
        equiv(x + 1) == (1 + x) -> True
        equiv(x + 1) == equiv(1 + x) -> True
    """
    if isinstance(expr, Equivalent):
        return expr

    # TODO: Add a ComparisonModifier class or some such Exact and Equivalent inherit from.
    if hasattr(expr, "expr"):
        expr = expr.expr
    return Equivalent(Expression.wrap(expr))


def is_equivalent(a: Expression, b: Expression) -> bool:
    """TODO: Crazy stuff happens here.

    Requires:
     - Valid Knowledgebase from knowledgebase.the_knowledgebase()
    """
    kbase = the_knowledgebase()

    with kbase.block_writes():
        for match in kbase.query(a):
            if match.rhs.sub(match.mapping) == exact(b):
                return True

    return False


# TODO: Delete
def __binary_equiv(self, other) -> bool:
    if isinstance(other, Expression) and self._hash_() == other._hash_():
        return True

    if isinstance(other, type(self)):

        def is_equal(a: Expression, b: Expression):
            return a == equiv(b)

        def reduce_constants(a: Constant, b: Constant):
            result = type(self).apply(a, b)
            assert isinstance(result, Constant)
            return result

        self_chain = combine_constants(self._extract_chain(), self._can_swap, reduce_constants)
        other_chain = combine_constants(other._extract_chain(), self._can_swap, reduce_constants)

        return equivalent_chains(self_chain, other_chain, is_equal, self._can_swap)

    return NotImplemented
