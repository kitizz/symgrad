# Copyright, Christopher Ham, 2022

from __future__ import annotations

from abc import abstractstaticmethod
import logging
from typing import Any

from symgrad.shared_context import thread_local_context

from ..expression import Expression
from ..set import Set, Null
from .operator import Operator, short_var_name

__all__ = [
    "UnaryOperator",
]


class UnaryOperator(Operator):
    a: Expression
    __match_args__ = ("a",)

    #: Used to render this operator is code and debug strings.
    #: Child operator classes should define this.
    code: str

    @abstractstaticmethod
    def _apply(a):
        """Implementation for the underlying operator."""
        raise NotImplementedError("Must implement _apply method for UnaryOperator subclass")

    def _eval_(self, substitutions: dict[str, Any]):
        return self._apply(self.a._eval_(substitutions))

    null = Null()

    def __init__(self, a):
        self.a = self.wrap(a)

        a_set = self.a.output_set
        if a_set is self.null:
            raise ValueError(f"Cannot operate on incomplete Sets, ({a_set})")

        kbase = Expression._spaghetti().the_knowledgebase()
        output_set = kbase.query_unary_set_rule(type(self), a_set)
        if output_set is None:
            if not thread_local_context().defining_rules:
                raise ValueError(f"Operator, {type(self).__name__}, does not operate on ({a_set})")
            output_set = Null()

        op_name = type(self).__name__.lower()
        name = f"{op_name}_{short_var_name(self.a)}"

        super().__init__(name, output_set)

        self.variables = self.a.variables
        self.operator_count = self.a.operator_count + 1
        self.operand_count = self.a.operand_count
        self.is_constant = self.a.is_constant
        if self.is_constant:
            self.constant_val = self._apply(self.a.constant_val)

    def __init_subclass__(cls):
        # Each subclass has its own _rules dict.
        assert hasattr(cls, "code")
        super().__init_subclass__()

    def _hash_(self) -> int:
        if self._expr_hash is None:
            # Cache the hash, since Generators are functionally immutable.
            self._expr_hash = hash((type(self).__name__, self.a._hash_()))

        return self._expr_hash

    def _str_debug(self, parent: Expression | None):
        out = self.code.format(a=self.a._str_debug(self))
        out = f"({out})"
        return out

    def _str_display(self, parent: Expression | None):
        out = self.code.format(a=self.a._str_display(self))
        out = f"({out})"
        return out
