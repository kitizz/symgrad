# Copyright 2022, Christopher Ham.

import logging
from typing import Any

from .expression import Expression
from .set import Set

__all__ = ["Constant"]


class Constant(Expression):
    """Constants have an explicit value and may be automatically combined/simplified.

    A Constant (like any Expression type) belongs to a known mathematical set
    (also known as output_set).
    """

    def __init__(self, value, output_set: Set | None = None):
        output_set = output_set or Set.find(value)
        super().__init__(str(value), output_set=output_set)
        self.value = value
        self.operand_count = 1
        self.is_constant = True
        self.constant_val = self.value

    def __getitem__(self, sl):
        raise RuntimeError("I don't know if this should be here...")

    def __eq__(self, other) -> bool:
        if isinstance(other, Constant):
            return self.value == other.value
        elif not isinstance(other, Expression):
            return self.value == other
        return NotImplemented

    def _hash_(self):
        if not hasattr(self, "_const_hash"):
            self._const_hash = hash(str(self.value))

        return self._const_hash

    def _eval_(self, substitutions: dict[str, Any]):
        if self.value is None:
            raise NotImplementedError("Need to figure out how to feed in applicable identities.")
        return self.value

    def _str_debug(self, parent: Expression | None):
        # Avoid multiline messiness.
        value_str = str(self.value).split("\n")[0]
        return f"{self.output_set._str_debug()}({value_str})"

    def _str_display(self, parent: Expression | None):
        if self.value is not None:
            # Avoid multiline messiness.
            return str(self.value).split("\n")[0]
        else:
            return self._str_debug(parent)
