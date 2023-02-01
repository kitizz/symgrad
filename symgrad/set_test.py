# Copyright, Christopher Ham, 2022

import logging

import pytest

from symgrad.operators.binary_operator import BinaryRuleNotFoundError
from symgrad.rules.knowledgebase import the_knowledgebase
from symgrad.shared_context import define_rules, run_test_with_knowledgebase

from .expression import Expression
from .set import Set
from .variable import Variable


class NoArgSet(Set):
    ...


class ArgOneSet(Set):
    dim: int


class ArgTwoSet(Set):
    dtype: Set
    dim: int


class Array3d(Set):
    rows: int
    cols: int
    stacks: int


class Array2d(Set):
    rows: int
    cols: int

    @classmethod
    def _supersets(cls):
        return {Array3d: (cls.rows, cls.cols, 1)}


class Array1d(Set):
    rows: int

    @classmethod
    def _supersets(cls):
        return {Array2d: (cls.rows, 1)}


#
# Tests below
#


def test_no_arg_errors():
    NoArgSet()

    with pytest.raises(TypeError):
        NoArgSet(5)

    with pytest.raises(TypeError):
        NoArgSet(a=5)


def test_one_arg_errors():
    ArgOneSet(1)
    ArgOneSet(dim=1)

    with pytest.raises(TypeError):
        ArgOneSet()

    with pytest.raises(TypeError):
        ArgOneSet(a=5)

    with pytest.raises(TypeError):
        ArgOneSet(5, 10)

    with pytest.raises(TypeError):
        ArgOneSet(5, dim=5)

    with pytest.raises(TypeError):
        ArgOneSet(dim=5, b=4)


def test_singleton_no_arg():
    a = NoArgSet()
    b = NoArgSet()
    c = ArgOneSet(5)

    assert a is b
    assert a is not c


def test_singleton_one_arg():
    a = ArgOneSet(5)
    b = ArgOneSet(dim=5)
    c = ArgOneSet(dim=6)

    assert a is b
    assert a is not c


def test_singleton_two_arg():
    a = ArgTwoSet(NoArgSet(), 5)
    b = ArgTwoSet(dim=5, dtype=NoArgSet())
    c = ArgTwoSet(dtype=NoArgSet(), dim=5)
    d = ArgTwoSet(NoArgSet(), dim=6)

    assert a is b
    assert a is c
    assert a is not d


def test_subsets():
    assert Array3d(2, 3, 4) in Array3d(2, 3, 4)
    assert Array3d(1, 3, 4) not in Array3d(2, 3, 4)
    assert Array3d(2, 4, 4) not in Array3d(2, 3, 4)
    assert Array3d(2, 3, 5) not in Array3d(2, 3, 4)
    assert Array3d(4, 3, 2) not in Array3d(2, 3, 4)

    assert Array2d(3, 2) in Array3d(3, 2, 1)
    assert Array2d(2, 2) not in Array3d(3, 2, 1)


def test_symbol_subset():
    x = Variable("x", NoArgSet())
    y = Variable("y", ArgTwoSet(NoArgSet(), 5))
    arr2 = Variable("a", Array2d(2, 3))
    arr3 = Variable("b", Array3d(2, 3, 1))

    assert x in NoArgSet()
    assert y in ArgTwoSet(NoArgSet(), 5)

    assert x not in ArgTwoSet(NoArgSet(), 5)
    assert y not in NoArgSet()

    assert arr2 in Array2d(2, 3)
    assert arr2 in Array3d(2, 3, 1)
    assert arr3 in Array3d(2, 3, 1)
    assert arr3 not in Array2d(2, 3)


def test_is_superset_class():
    arr1 = Array1d(3)
    arr2 = Array2d(4, 5)
    arr3 = Array3d(6, 7, 8)

    assert arr1.is_superset_class(Array1d)
    assert arr1.is_superset_class(Array2d)
    assert arr1.is_superset_class(Array3d)

    assert not arr2.is_superset_class(Array1d)
    assert arr2.is_superset_class(Array2d)
    assert arr2.is_superset_class(Array3d)

    assert not arr3.is_superset_class(Array1d)
    assert not arr3.is_superset_class(Array2d)
    assert arr3.is_superset_class(Array3d)


def test_params():
    arr2 = Array2d(4, 5)
    arr3 = Array3d(6, 7, 8)

    assert arr2.params() == {"rows": 4, "cols": 5}
    assert arr3.params() == {"rows": 6, "cols": 7, "stacks": 8}


def test_supersets():
    arr1 = Array1d(3)
    arr2 = Array2d(3, 1)
    arr3 = Array3d(3, 1, 1)

    assert set(arr1.supersets()) == {arr1, arr2, arr3}
    assert set(arr2.supersets()) == {arr2, arr3}
    assert set(arr3.supersets()) == {arr3}


def test_operator_inputs():
    x = Variable("x", NoArgSet())
    y = Variable("y", NoArgSet())

    # Trying to add x and y should raise an error before trying to define how adding them works.
    with pytest.raises(BinaryRuleNotFoundError):
        x + y

    kbase = the_knowledgebase()
    with run_test_with_knowledgebase(kbase), define_rules():
        (x + y) in NoArgSet()

    assert (x + y).output_set is NoArgSet()


class SetLevel1a(Set):
    ...


class SetLevel1b(Set):
    ...


class SetLevel2(Set):
    @classmethod
    def _supersets(cls):
        return {SetLevel1a: ()}


class SetLevel3(Set):
    @classmethod
    def _supersets(cls):
        return {SetLevel2: ()}


def test_superset_distance():
    l1a = SetLevel1a()
    l2 = SetLevel2()
    l3 = SetLevel3()

    assert l1a.superset_distance(l1a) == 0
    assert l2.superset_distance(l2) == 0
    assert l3.superset_distance(l3) == 0

    assert l2.superset_distance(l1a) == 1
    assert l3.superset_distance(l2) == 1

    assert l3.superset_distance(l1a) == 2

    assert l1a.superset_distance(l3) == -1
