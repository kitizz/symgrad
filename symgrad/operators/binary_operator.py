# Copyright, Christopher Ham, 2022

from __future__ import annotations

import functools
import logging
from abc import abstractstaticmethod
from collections.abc import Iterable
from typing import Any

from ..expression import Expression
from ..set import Null, Set
from ..shared_context import thread_local_context
from ..variable import Variable
from .operator import Operator, short_var_name

__all__ = [
    "BinaryOperator",
    "BinaryRuleNotFoundError",
]


class BinaryRuleNotFoundError(Exception):
    ...


class BinaryOperator(Operator):
    # TODO: Make BinaryOperator Generic on the a, b types once default Generic args are allowed.
    a: Expression
    b: Expression
    __match_args__ = ("a", "b")

    #: Used to render this operator is code and debug strings.
    #: Child operator classes should define this.
    code: str

    def __new__(cls, a, b):
        op = super().__new__(cls)
        op._init(a, b)

        # TODO: Try to simplify the expression here...

        return op

    def __init__(self, a, b):
        ...

    null = Null()

    def _init(self, a, b):
        self.a = self.wrap(a)
        self.b = self.wrap(b)

        a_set = self.a.output_set
        b_set = self.b.output_set
        if a_set is self.null or b_set is self.null:
            raise BinaryRuleNotFoundError(f"Cannot operate on incomplete Sets, ({a_set}, {b_set})")

        kbase = Expression._spaghetti().the_knowledgebase()
        output_set = kbase.query_binary_set_rule(type(self), a_set, b_set)
        if output_set is None:
            if not thread_local_context().defining_rules:
                raise BinaryRuleNotFoundError(
                    f"Operator, {type(self).__name__}, does not operate on ({a_set}, {b_set})"
                )
            output_set = Null()

        op_name = type(self).__name__.lower()
        a_name = short_var_name(self.a, hex_len=6)
        b_name = short_var_name(self.b, hex_len=6)
        name = f"{op_name}_{a_name}_{b_name}"

        super().__init__(name, output_set)

        self.variables = _union_variables(self.a.variables, self.b.variables)
        self.operator_count = self.a.operator_count + self.b.operator_count + 1
        self.operand_count = self.a.operand_count + self.b.operand_count
        self.is_constant = self.a.is_constant and self.b.is_constant
        if self.is_constant:
            self.constant_val = self._apply(self.a.constant_val, self.b.constant_val)

    @classmethod
    def reduce(cls, chain: Iterable[Expression], initial=None) -> Expression:
        """Reduce this BinaryOperator of a chain of Expressions.

        Requires:
         - len(chain) > 0 when initial is None (undefined)
         - initial can be wrapped by Expression if defined

        Ensures:
         - This BinaryOperator is apply accumulatively to the input chain from left to right.

        Example:
          Add.reduce([1, x, 9]) == x + 10
        """
        if initial is not None:
            return functools.reduce(cls, chain, Expression.wrap(initial))
        else:
            return functools.reduce(cls, chain)

    @abstractstaticmethod
    def _apply(a, b):
        """Implementation for the underlying operator."""
        raise NotImplementedError("Must implement _apply method for BinaryOperator subclass")

    def _eval_(self, substitutions: dict[str, Any]):
        return self._apply(self.a._eval_(substitutions), self.b._eval_(substitutions))

    def __init_subclass__(cls):
        # Each subclass has its own _rules dict.
        assert hasattr(cls, "code")
        super().__init_subclass__()

    def _hash_(self) -> int:
        if self._expr_hash is None:
            # Cache the hash, since Generators are functionally immutable.
            self._expr_hash = hash((type(self).__name__, self.a._hash_(), self.b._hash_()))

        return self._expr_hash

    def _str_debug(self, parent: Expression | None):
        out = self.code.format(a=self.a._str_debug(self), b=self.b._str_debug(self))
        out = f"({out})"
        return out

    def _str_display(self, parent: Expression | None):
        out = self.code.format(a=self.a._str_display(self), b=self.b._str_display(self))
        out = f"({out})"
        return out

    @classmethod
    def _sort_key(cls, expr: Expression) -> tuple:
        """In a chain of repeated applications of the same BinaryOperator, sort
        the terms in ascending order of this key.

        Override this in operator implementations for different behavior.
        """
        return (expr.operator_count, len(expr.variables), expr.name)


def _union_variables(
    a_vars: dict[str, Variable], b_vars: dict[str, Variable]
) -> dict[str, Variable]:
    """Combine two sets of Variables while looking out for differing output Set types."""
    new_vars = a_vars.copy()
    for b_var in b_vars.values():
        existing = new_vars.setdefault(b_var.name, b_var)
        if existing._hash_() != b_var._hash_():
            raise TypeError(
                "Variables sharing the same name, but with differing types may "
                "not appear in the same Expression. "
                f"Found '{b_var.name}' with types {existing.output_set} and {b_var.output_set}."
            )

    return new_vars
