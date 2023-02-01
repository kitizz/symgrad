# Copyright, Christopher Ham, 2022

from .unary_operator import UnaryOperator


__all__ = ["Inverse"]


class Inverse(UnaryOperator):
    code = "{a}**-1"
