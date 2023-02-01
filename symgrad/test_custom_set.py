# Copyright, Christopher Ham, 2022

"""Test whether one can define a custom set and its rules to operate on.

These tests focus on the classic 90deg rotation and flip symmetric group.
"""

from enum import Enum

# TODO: Remove
class Set:
    ...


class Symbol:
    ...


class Rule:
    ...


class RotateFlipState(Enum):
    IDENTITY = 0
    ROTATE_90 = 1
    ROTATE_180 = 2
    ROTATE_270 = 3
    FLIP_0 = 4
    FLIP_45 = 5
    FLIP_90 = 6
    FLIP_135 = 7


def compose_rotate_flip(x: RotateFlipState, y: RotateFlipState) -> RotateFlipState:
    S = RotateFlipState
    rots = (S.IDENTITY, S.ROTATE_90, S.ROTATE_180, S.ROTATE_270)
    flips = (S.FLIP_0, S.FLIP_45, S.FLIP_90, S.FLIP_135)

    if x == S.IDENTITY:
        return y
    elif y == S.IDENTITY:
        return x

    # TODO: Validate this logic...
    if x in rots and y in rots:
        # Forms a mod 4 group:
        x_ind = rots.index(x)
        y_ind = rots.index(y)
        return rots[(x_ind + y_ind) % 4]

    if x in flips and y in rots:
        # Forms a mod 4 group:
        x_ind = flips.index(x)
        y_ind = rots.index(y)
        return flips[(x_ind + y_ind) % 4]

    if x in flips and y in flips:
        x_ind = flips.index(x)
        y_ind = flips.index(y)
        diff = (y_ind - x_ind) % 4
        return rots[diff]

    if x in rots and y in flips:
        x_ind = rots.index(x)
        y_ind = flips.index(y)
        return flips[(x_ind + y_ind) % 4]


class RotationsAndFlips(Set):
    @classmethod
    def validate(cls, x) -> bool:
        return isinstance(x, RotateFlipState)

    @classmethod
    def sample(cls, size=None, seed=None) -> RotateFlipState:
        return [state for state in RotateFlipState]


class Rotations(RotationsAndFlips):
    _set = {
        RotateFlipState.IDENTITY,
        RotateFlipState.ROTATE_90,
        RotateFlipState.ROTATE_180,
        RotateFlipState.ROTATE_270,
    }

    @classmethod
    def validate(cls, x):
        return x in cls._set

    @classmethod
    def sample(cls, size=None, seed=None) -> RotateFlipState:
        return list(cls._set)


# Register rules for how these can be composed...
# class RotateFlipMultiply(Rule):
#     operator = Multiply
#     inputs = (RotationsAndFlips, RotationsAndFlips)
#     output = RotationsAndFlips
#     identity = RotateFlipState.IDENTITY
#     associative = True
#     commutative = False

#     @classmethod
#     def compose(cls, x: RotateFlipState, y: RotateFlipState) -> RotateFlipState:
#         return compose_rotate_flip(x, y)


# register_rule(
#     operator=Multiply,
#     inputs=(Rotations, Rotations),
#     output=Rotations,
#     identity=RotateFlipState.IDENTITY,
#     associative=True,
#     commutative=False,
#     compose=compose_rotate_flip,
# )


def test_sets():
    S = RotateFlipState  # Make readable.

    rots = (S.IDENTITY, S.ROTATE_90, S.ROTATE_180, S.ROTATE_270)
    flips = (S.FLIP_0, S.FLIP_45, S.FLIP_90, S.FLIP_135)

    for rot in rots:
        rot_gen = make_generator(rot)
        assert rot_gen.output_type() is Rotations
        assert rot_gen in Rotations
        assert rot_gen in RotationsAndFlips

    for flip in flips:
        flip_gen = make_generator(flip)
        assert flip_gen.output_type() is RotationsAndFlips
        assert flip_gen not in Rotations
        assert flip_gen in RotationsAndFlips


def test_symbols():
    x = Symbol("x", set=RotationsAndFlips)

    S = RotateFlipState  # Make readable.
    assert x.output_type() is RotationsAndFlips
    assert (x * S.ROTATE_90).output_type() is RotationsAndFlips

    assert x * S.IDENTITY == x
    assert x * S.ROTATE_90 == x * S.ROTATE_90
    assert x * S.ROTATE_90 * S.ROTATE_90 == x * S.ROTATE_180


def test_upcasting():
    x = Symbol("x", set=RotationsAndFlips)
    y = Symbol("y", set=Rotations)

    assert x.outpute_type() is RotationsAndFlips
    assert y.outpute_type() is Rotations
    assert (x * x).output_type() is RotationsAndFlips
    assert (y * y).output_type() is Rotations
    assert (x * y).output_type() is RotationsAndFlips
    assert (y * x).output_type() is RotationsAndFlips


def test_power():
    x = Symbol("x", set=RotationsAndFlips)

    S = RotateFlipState  # Make readable.
    assert x * S.ROTATE_90**2 == x * S.ROTATE_180
