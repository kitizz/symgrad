# Copyright, Christopher Ham, 2022

import logging

from ..expression import Expression
from .binary_operator import BinaryOperator


__all__ = ["Multiply"]


class Multiply(BinaryOperator):
    code = "{a} * {b}"

    @staticmethod
    def _apply(a, b):
        return a * b

    @classmethod
    def _sort_key(cls, expr: Expression) -> tuple:
        """Tries to keep Constants before Variables, and more complex terms at the end."""
        var_name = next(iter(expr.variables)) if expr.variables else ""
        return (var_name, expr.operator_count, expr.name)
