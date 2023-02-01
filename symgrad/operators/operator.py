# Copyright, Christopher Ham, 2022

from __future__ import annotations

from ..expression import Expression


__all__ = [
    "Operator",
]


class Operator(Expression):
    def __init_subclass__(cls):
        cls._operators[cls.__name__] = cls

    @classmethod
    def add_rule(cls, **kwargs):
        """Overload this in child Operators"""
        ...


def short_var_name(g: Expression, max_len=10, hex_len=8):
    if len(g.name) <= max_len:
        return g.name
    else:
        # Construct a reasonably readable name from hashes.
        return format(g._hash_(), "x")[:hex_len]
