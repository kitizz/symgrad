# Copyright, Christopher Ham, 2022

"""

For now, we hold back on defining an ultimate Tensor set since until a great
need arises. Some challenges remain around good type annotations.

## Caveats!
The definitions of Matrix and Vector here may be skewed towards the computer-
science side of things. For an excellent resource that tries to describe the
strict mathematical definitions of these sets/objects, see:
https://math.stackexchange.com/a/1817190
"""


from __future__ import annotations

import logging

import numpy as np

from ..set import Set
from .set_type import SetType
from .numbers import Reals, Ints
from ..variable import Variable

__all__ = [
    "Matrix",
    "IdentityMatrix",
]


class Matrix(Set):
    dtype: Set
    rows: int
    cols: int

    def __init__(self, dtype: Set, rows: int, cols: int = 1):
        assert dtype in SetType(upper_bound=Reals())
        # TODO: More generally, dtype should have addition and multiplication defined.
        # assert Add.find_rule(dtype, dtype) is not None
        # assert Multiply.find_rule(dtype, dtype) is not None
        self.dtype = dtype
        self.rows = rows
        self.cols = cols

    # @classmethod
    def validate(self, value) -> bool:
        if not isinstance(value, np.ndarray):
            return False

        # TODO: How to get these values?
        if value.shape != (self.rows, self.cols):
            return False

        return self.dtype.validate(value.flat[0])


# TODO: Finally delete? Strictly speaking, in maths, anything that forms a vector-space is a vector.
# So maybe we should leave it out for now until the implications are better understood.
# class Vector(Matrix):
#     dtype: Set
#     rows: int

#     def __init__(self, dtype: Set, rows: int):
#         super().__init__(dtype, rows, 1)

#     def _supersets(self) -> tuple[Set, ...]:
#         return (Matrix.like(self.dtype, self.rows, 1),)


class IdentityMatrix(Matrix):
    def __init__(self, dtype: Set, size: int):
        super().__init__(dtype, size, size)

    def _supersets(self):
        return (Matrix.like(self.dtype, self.rows, self.cols),)


class ZeroMatrix(Matrix):
    def __init__(self, dtype: Set, rows: int, cols: int):
        super().__init__(dtype, rows, cols)

    def _supersets(self):
        return (Matrix.like(self.dtype, self.rows, self.cols),)
