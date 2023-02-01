# Copyright 2022, Christopher Ham

from .constant import Constant
from .set_element import SetElement

__all__ = [
    "wrap_constant",
]


def wrap_constant(value) -> Constant | None:
    wrapped = SetElement.find(value)
    if wrapped is None:
        wrapped = Constant(value)

    return wrapped
