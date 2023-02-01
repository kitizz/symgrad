# Copyright, Christopher Ham, 2022

import logging
from .binary_operator import BinaryOperator
from ..expression import Expression


__all__ = ["Add"]


class Add(BinaryOperator):
    code = "{a} + {b}"

    @staticmethod
    def _apply(a, b):
        return a + b

    @classmethod
    def _sort_key(cls, expr: Expression) -> tuple:
        """Sort lexigraphically by (Complexity, Number of Variables, Name)"""
        var_name = next(iter(expr.variables)) if expr.variables else chr(ord("z") + 1)
        return (var_name, -expr.operator_count, expr.name)
