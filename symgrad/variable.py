# Copyright 2022, Christopher Ham.

from typing import Any

from .expression import Expression
from .set import Set


__all__ = ["Variable"]


class Variable(Expression):
    """Variables have no explicit value, and a unique name to identify them.

    A Constant (like any Expression type) belongs to a known mathematical set
    (also known as output_set).
    """

    def __init__(self, name: str, output_set: Set):
        super().__init__(name, output_set=output_set)
        self.variables = {name: self}
        self.operand_count = 1
        self.is_constant = False

    def __hash__(self):
        return self._hash_()

    def _eval_(self, substitutions: dict[str, Any]):
        value = substitutions.get(self.name, self)
        return getattr(value, "value", value)

    def _hash_(self):
        if not hasattr(self, "_var_hash"):
            self._var_hash = hash((self.output_set, self.name))
        return self._var_hash

    def _str_debug(self, parent: Expression | None):
        return f"{self.name}[{self.output_set}]"

    def _str_display(self, parent: Expression | None):
        return self.name


Expression._add_spaghetti("Variable", Variable)
