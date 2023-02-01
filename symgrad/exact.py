# Copyright, Christopher Ham, 2022

import logging
from typing import Any, Sequence

from .expression import Expression
from .constant import Constant
from .util import is_unordered_equal

__all__ = [
    "exact",
]


class Exact:
    expr: Expression

    def __init__(self, expr: Expression):
        self.expr = expr

    def __eq__(self, other):
        """Equality testing in the reverse order relies on Exact being a different
        type from any other Generator. And other Generators using _hash() as an initial check.
        """
        if isinstance(other, Expression):
            return self.expr._hash_() == other._hash_()

        if isinstance(other, Exact):
            return self.expr._hash_() == other.expr._hash_()

        if isinstance(self.expr, Constant):
            return self.expr.value == other

        return NotImplemented

    def __hash__(self):
        return self.expr._hash_()

    def __str__(self):
        return f"Exact({str(self.expr)})"

    def __repr__(self):
        return f"Exact({repr(self.expr)})"


class ExactUnordered:
    exprs: list[Exact]

    def __init__(self, exprs: Sequence[Expression]):
        self.exprs = [exact(expr) for expr in exprs]

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented

        return is_unordered_equal(self.exprs, other)

    def __str__(self):
        return f"ExactUnordered({tuple(str(v.expr) for v in self.exprs)})"

    def __repr__(self):
        return f"ExactUnordered({tuple(repr(v.expr) for v in self.exprs)})"


def exact_seq(exprs: Sequence[Any]) -> list[Exact]:
    return [exact(v) for v in exprs]


def exact_unordered(exprs: Sequence[Any]) -> ExactUnordered:
    return ExactUnordered(exprs)


def exact(expr: Any) -> Exact:
    """Ensures comparisons with other Generators compare that the expression matches exactly.
    As opposed to merely being equivalent.

    Examples:
        exact(x + 1) == (x + 1) -> True
        exact(x + 1) == exact(x + 1) -> True

        (x + 1) == exact(1 + x) -> False
        exact(x + 1) == (1 + x) -> False
        exact(x + 1) == exact(1 + x) -> False
    """
    if isinstance(expr, Exact):
        return expr

    if hasattr(expr, "expr"):
        expr = expr.expr
    return Exact(Expression.wrap(expr))
