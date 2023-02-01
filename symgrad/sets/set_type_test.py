# Copyright, Christopher Ham, 2022

from .set_type import SetType
from ..set import Set
from .numbers import Reals
from ..variable import Variable


class FooBar(Set):
    ...


class Foo(FooBar):
    ...


class Bar(Set):
    def _supersets(self) -> dict[type[Set], tuple]:
        return {FooBar: ()}


def test_contains_self():
    assert SetType() in SetType()
    assert SetType(FooBar()) in SetType()
    assert SetType(Foo()) in SetType()
    assert SetType(Bar()) in SetType()

    assert SetType() not in SetType(FooBar())
    assert SetType(FooBar()) in SetType(FooBar())
    assert SetType(Foo()) not in SetType(FooBar())
    assert SetType(Bar()) in SetType(FooBar())


def test_contains_symbol():
    T = Variable("T", SetType(upper_bound=Reals()))

    assert T in SetType(upper_bound=Reals())
