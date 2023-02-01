# Copyright, Christopher Ham, 2022

from typing import Any
from .binary_operator import BinaryOperator


__all__ = ["Power"]


class Power(BinaryOperator):
    code = "{a}**{b}"

    @staticmethod
    def _apply(a, b):
        return a**b
