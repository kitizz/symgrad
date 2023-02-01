# Copyright, Christopher Ham, 2022

from ..binary_operator import *
from ..expression import Expression
from ..symbol import *
from ..unary_operator import *


__all__ = [
    "expand",
]


def expand(gen: Expression, max_recursion=100) -> Expression:
    # gen = simplify(gen)
    for i in range(max_recursion):
        next_gen = _expand_recursive(gen, first=True)
        if next_gen.name == gen.name:
            return gen
        gen = next_gen
    else:
        raise RuntimeError("Expansion exceed max recursion limit")


def _expand_recursive(gen: Expression, first=False) -> Expression:
    """Expand multiplication and division of sums."""
    op = gen.__class__
    op_gen = getattr(gen, "from_args", None)

    match gen:
        case Multiply(a, b):
            a_r = _expand_recursive(a)
            b_r = _expand_recursive(b)

            new_gen = op_gen(a_r, b_r)

            match a_r:
                case Add(a1, a2):
                    # print(f"Expand ({a1} +- {a2}) */ {b_r}")
                    op_a = a_r.from_args
                    new_gen = op_a(simplify(op_gen(a1, b_r)), simplify(op_gen(a2, b_r)))

            match b_r:
                case Add(b1, b2):
                    # print(f"Expand ({a_r} */ {b1} +- {b2})")
                    op_b = b_r.from_args
                    new_gen = op_b(simplify(op_gen(a_r, b1)), simplify(op_gen(a_r, b2)))

            # return new_gen

        case ElementWisePower(Add() as base, CInt() as pwr):
            add_chain = base._extract_chain()

            out_chain = add_chain.copy()
            assert pwr.value >= 2
            for i in range(pwr.value - 1):
                new_chain = []
                for elem_a in add_chain:
                    for elem_b in out_chain:
                        new_chain.append(elem_a * elem_b)
                out_chain = new_chain
            new_gen = sum(out_chain)
            # num_coeffs = pwr + 1
            # coeffs = [scipy.special.comb(pwr, k) for k in range(num_coeffs)]
            # one_down = _expand_recursive(base ** (pwr - 1))
            # new_gen = a * one_down + b * one_down
            # print("Match power:", new_gen)
            # print(new_gen)

        case BinaryOperator(a, b):
            new_gen = op_gen(_expand_recursive(a), _expand_recursive(b))

        case UnaryOperator(a):
            new_gen = op_gen(_expand_recursive(a))

        case Symbol():
            # Can't expand any further.
            new_gen = gen
        case _:
            raise ValueError(f"Not sure what to do with type {op}")

    return simplify(new_gen)
