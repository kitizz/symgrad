# Copyright, Christopher Ham, 2022

from collections import defaultdict

import pytest

from ..set import Set


class BaseSet(Set):
    """Resetable specialization of the Set class specifically to use for tests."""

    @classmethod
    def reset(cls):
        cls._set_mappings = defaultdict(list)


@pytest.fixture(scope="function")
def default_set():
    BaseSet.reset()
    yield BaseSet
    BaseSet.reset()


#
# Various Sets implementing BaseSet
#


class NoArgSet(BaseSet):
    ...


class NoArgSubSet(BaseSet):
    @classmethod
    def _supersets(cls):
        return {NoArgSet: ()}


class OneArgSet(BaseSet):
    size: int


#
# Set mappings
#


def map_int_noarg(x: int) -> NoArgSet:
    return NoArgSet()


def map_int_onearg(x: int) -> OneArgSet:
    return OneArgSet(x)


def map_int_noargsub(x: int) -> NoArgSubSet | None:
    if x == 0:
        return NoArgSubSet()


#
# Actual Tests
#


def test_register_simple(default_set):
    BaseSet._register_mapping(map_int_noarg)

    assert BaseSet.find(2) is NoArgSet()
    assert BaseSet.find(0) is NoArgSet()
    assert BaseSet.find(-1) is NoArgSet()


def test_find_with_set(default_set):
    BaseSet._register_mapping(map_int_noarg)

    assert BaseSet.find(NoArgSet) is NoArgSet()
    assert BaseSet.find(NoArgSet()) is NoArgSet()


def test_find_with_type(default_set):
    BaseSet._register_mapping(map_int_noarg)

    assert BaseSet.find(int) is NoArgSet()


def test_find_with_type_multi(default_set):
    BaseSet._register_mapping(map_int_noarg)
    BaseSet._register_mapping(map_int_onearg)

    assert BaseSet.find(int) is NoArgSet()


# The ordering shouldn't matter in this case.
def test_find_with_type_multi_reversed(default_set):
    BaseSet._register_mapping(map_int_onearg)
    BaseSet._register_mapping(map_int_noarg)

    assert BaseSet.find(int) is NoArgSet()


def test_find_superset(default_set):
    BaseSet._register_mapping(map_int_noarg)
    BaseSet._register_mapping(map_int_noargsub)

    assert BaseSet.find(int) is NoArgSet()
    assert BaseSet.find(1) is NoArgSet()
    assert BaseSet.find(0) is NoArgSubSet()


def test_find_superset_reversed(default_set):
    BaseSet._register_mapping(map_int_noargsub)
    BaseSet._register_mapping(map_int_noarg)

    assert BaseSet.find(int) is NoArgSet()
    assert BaseSet.find(1) is NoArgSet()
    assert BaseSet.find(0) is NoArgSubSet()


def test_find_ambiguous(default_set):
    # The ambiguous case should always choose the last registered mapping
    # Amgiguous because NoArgSet and OneArgSet are not related by containement.
    BaseSet._register_mapping(map_int_noarg)
    BaseSet._register_mapping(map_int_onearg)

    assert BaseSet.find(0) is OneArgSet(0)


def test_find_ambiguous_reversed(default_set):
    # The ambiguous case should always choose the last registered mapping
    # Amgiguous because NoArgSet and OneArgSet are not related by containement.
    BaseSet._register_mapping(map_int_onearg)
    BaseSet._register_mapping(map_int_noarg)

    assert BaseSet.find(0) is NoArgSet()
