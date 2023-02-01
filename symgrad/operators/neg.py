# Copyright, Christopher Ham, 2022

from typing import Any
from .unary_operator import UnaryOperator


__all__ = ["Neg"]


class Neg(UnaryOperator):
    code = "-{a}"

    @staticmethod
    def _apply(a):
        return -a
