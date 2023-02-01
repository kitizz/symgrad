# Copyright, Christopher Ham, 2022

from typing import Any

from symgrad.variable import Variable

from ..operators import Add, Inverse, Multiply, Neg, Power
from ..rules.rule_collection import RuleCollection
from ..set import Set, register_set_mapping
from ..set_element import SetElement, SetElementMapping

__all__ = [
    "Reals",
    "Ints",
    "NonNegativeInts",
    "Zero",
    "One",
]


class Reals(Set):
    ...


class Ints(Set):
    @classmethod
    def _supersets(cls):
        return {Reals: ()}


class NonNegativeInts(Set):
    @classmethod
    def _supersets(cls) -> dict[type[Set], tuple]:
        return {Ints: ()}


class Zero(SetElement):
    def _str_display(self, parent) -> str:
        return "0"


class One(SetElement):
    def _str_display(self, parent) -> str:
        return "1"


@register_set_mapping
def real_from_float(x: float) -> Reals:
    return Reals()


@register_set_mapping
def int_from_int(x: int) -> Ints:
    return Ints()


@register_set_mapping
def int_from_float(x: float) -> Ints | None:
    return Ints() if x.is_integer() else None


@register_set_mapping
def non_neg_int_from_int(x: int) -> NonNegativeInts | None:
    return NonNegativeInts() if x >= 0 else None


@register_set_mapping
def non_neg_int_from_float(x: float) -> NonNegativeInts | None:
    return NonNegativeInts() if x.is_integer() and x >= 0 else None


class ZeroFloat(SetElementMapping):
    element = Zero
    set = Reals()
    static_value = 0.0


class OneFloat(SetElementMapping):
    element = One
    set = Reals()
    static_value = 1.0


class ZeroInt(SetElementMapping):
    element = Zero
    set = NonNegativeInts()
    static_value = 0


class OneInt(SetElementMapping):
    element = One
    set = NonNegativeInts()
    static_value = 1


class NumberRules(RuleCollection):
    @classmethod
    def axioms(cls):
        x = Variable("x", Reals())
        y = Variable("y", Reals())
        z = Variable("z", Reals())

        i = Variable("i", Ints())
        j = Variable("j", Ints())
        k = Variable("k", Ints())

        # Define the Set output rules.
        (x + y) in Reals()
        (x * y) in Reals()
        (i + j) in Ints()
        (i * j) in Ints()

        (x**y) in Reals()
        (x**i) in Reals()
        (i**x) in Reals()
        (i**j) in Ints()

        (-x) in Reals()
        (-i) in Ints()
        Inverse(x) in Reals()

        # Define axioms.
        # The Add group over Reals:
        (x + y) + z == x + (y + z)
        x + y == y + x
        x + 0 == x
        x - x == 0

        # The Multiply group over Reals
        (x * y) * z == x * (y * z)
        x * y == y * x
        x * 1 == x
        x / x == 1

        # Distributive
        x * (y + z) == (x * y) + (x * z)
        x ** (y + z) == (x**y) * (x**z)
        (x * y) ** z == (x**z) * (y**z)


    @classmethod
    def theorems(cls):
        x = Variable("x", Reals())
        y = Variable("y", Reals())
        z = Variable("z", Reals())
        
        # TODO: Autogenerate these at some point...
        (y + z) * x == (y * x) + (z * x)
        x**1 == x
        x ** (-1) == Inverse(x)
        
        # Negatives
        -x == (-1) * x
        -(x * y) == (-x) * y
        -(x * y) == x * (-y)
        -(-x) == x
        -x == -1 * x
        (-x) * (-y) == x * y

        x**1 == x
        x**0 == 1
        x * x == x**2
        (x**y) * x == x ** (y + 1)
        x * (x**y) == x ** (y + 1)
        (x**y) ** z == x ** (y * z)

        # Notice that this implicitly follows the order of arguments.
        x + x == 2 * x
        (x * y) + x == x * (y + 1)
        x + (x * y) == x * (y + 1)
        (y * x) + x == (y + 1) * x
        x + (y * x) == (y + 1) * x