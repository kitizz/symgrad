# Copyright, Christopher Ham, 2022

import itertools
import logging

from ..binary_operator import BinaryOperator, Group
from ..expression import Expression
from ..unary_operator import UnaryOperator
from ..symbol import Symbol

__all__ = [
    "cancel",
]


def cancel(gen: Expression) -> Expression:
    """Recursively cancel or reduce any terms in the input Generator.

    Some examples:
        Real (commutative):
        x * y * z / z -> x * y
        x * y * z / x -> y * z

        Matrix (non-commutative):
        X * Y * Z * (Z^-1) -> X * Y
        X * Y * Z * (X^-1) -> X * Y * Z * X^-1
    """

    match gen:
        case BinaryOperator():
            return _maybe_cancel_group(gen)

        case UnaryOperator(a):
            op_class = type(gen)
            return op_class(cancel(a))

        case Symbol():
            return gen

        case _:
            raise ValueError(f"Not sure what to do with type, {type(gen)}")


def _maybe_cancel_group(gen: BinaryOperator) -> BinaryOperator:
    """If the operands of this operator form a known Group, try to cancel any operands."""
    group = gen.get_group()
    if group is None:
        return gen

    # Remove any identity terms to simplify early.
    recursive_cancel = (cancel(v) for v in gen._extract_chain())
    operand_chain = [v for v in recursive_cancel if type(v) is not group.identity]

    # Attempt to cancel pairwise terms in operand chain.
    # While-loop lets us "refresh" the operand_chain iterator whenever a pair is cancelled.
    while True:
        if group.commutative:
            pair_iter = itertools.combinations(range(len(operand_chain)), r=2)
        else:
            pair_iter = itertools.pairwise(range(len(operand_chain)))

        pair_cancelled = False
        for i, j in pair_iter:
            if _can_cancel_terms(operand_chain[i], operand_chain[j], group):
                # Cancel this pair, and then loop back through the newly modified operand_chain.
                assert j > i
                del operand_chain[j]
                del operand_chain[i]
                pair_cancelled = True
                break

        if not pair_cancelled:
            # Nothing more found, tap out.
            break

    # Ensure the identity remains if everything cancels.
    operand_chain = operand_chain or [group.identity()]

    # Accumulate the chain back into a single generator.
    operator = type(gen)
    output = operand_chain[0]
    for v in operand_chain[1:]:
        output = operator(output, v)

    return output


def _can_cancel_terms(a: Expression, b: Expression, group: Group) -> bool:
    if type(a) is group.inverse == type(b) is group.inverse:
        # One and only one must be the inverse to be able to cancel these.
        return False

    if type(a) is group.inverse and b == a.a:
        return True

    if type(b) is group.inverse and a == b.a:
        return True

    return False
