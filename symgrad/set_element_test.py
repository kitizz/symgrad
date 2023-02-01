# Copyright

from dataclasses import dataclass

import numpy as np

from .set_element import SetElement, SetElementMapping
from .set import Set
from .variable import Variable
from .expression import Expression


# Main library defines theoretical types like so...


class Floats(Set):
    ...


class Ints(Set):
    ...


class Array2d(Set):
    # Implicitly float to simplify tests.
    rows: int
    cols: int


class OneElem(SetElement):
    ...


class ZeroElem(SetElement):
    ...


# Then concrete value mappings are defined separately like so...
# We define custom data types to avoid interferring with built-in types in the library state.
@dataclass
class Float:
    value: float


@dataclass
class Int:
    value: int


class ZeroFloat(SetElementMapping):
    element = ZeroElem
    set = Floats()

    @classmethod
    def from_symbol(cls, set: Floats) -> Float:
        return Float(0.0)


class OneFloat(SetElementMapping):
    element = OneElem
    set = Floats()

    @classmethod
    def from_symbol(cls, set: Floats) -> Float:
        return Float(1.0)


class ZeroInt(SetElementMapping):
    element = ZeroElem
    set = Ints()
    static_value = Int(0)


class OneInt(SetElementMapping):
    element = OneElem
    set = Ints()
    static_value = Int(1)


class OneDiagArray(SetElementMapping):
    N = Variable("N", Ints())

    element = OneElem
    set = Array2d(N, N)  # Pattern...
    value_type = np.ndarray

    @classmethod
    def from_symbol(cls, set: Array2d) -> np.ndarray | None:
        if set.rows == set.cols:
            return np.identity(set.rows)


def test_values():
    assert OneElem(Floats()).value == Float(1.0)
    assert OneElem(Ints()).value == Int(1)
    assert ZeroElem(Floats()).value == Float(0.0)
    assert ZeroElem(Ints()).value == Int(0)


def test_find_from_symbol():
    element = SetElement.find(Float(0.0))
    assert isinstance(element, ZeroElem)
    assert element.output_set is Floats()
    assert element.value == Float(0.0)

    element = SetElement.find(Float(1.0))
    assert isinstance(element, OneElem)
    assert element.output_set is Floats()
    assert element.value == Float(1.0)


def test_find_static_value():
    element = SetElement.find(Int(0))
    assert isinstance(element, ZeroElem)
    assert element.output_set is Ints()
    assert element.value == Int(0)

    element = SetElement.find(Int(1))
    assert isinstance(element, OneElem)
    assert element.output_set is Ints()
    assert element.value == Int(1)


def test_wrap():
    assert isinstance(Expression.wrap(Float(0.0)), ZeroElem)
    assert isinstance(Expression.wrap(Int(0)), ZeroElem)
    assert isinstance(Expression.wrap(Float(1.0)), OneElem)
    assert isinstance(Expression.wrap(Int(1)), OneElem)
