# Copyright 2022, Christopher Ham

from __future__ import annotations

from collections import defaultdict
from typing import Generic, TypeVar

from ..constant import Constant
from ..exact import exact
from ..expression import Expression

# from ..manipulation import expand
from ..operators import Add, BinaryOperator, Multiply, Power
from ..sets import Ints, Reals
from ..variable import Variable


__all__ = [
    "Polynomial",
    "ExtractionError",
    "PolynomialError",
    "extract_polynomial",
]

T = TypeVar("T")


class Polynomial:
    _coeffs: dict[int, Expression]
    _const_coeffs: list[float] | None
    degree: int

    def __init__(self, coefficients: dict[int, Expression]):
        self._coeffs = coefficients
        self.degree = max(self._coeffs.keys())

        all_coeffs_constant = True
        const_coeffs = [0.0] * (self.degree + 1)
        for exponent, coeff in self._coeffs.items():
            if not isinstance(coeff, Constant):
                all_coeffs_constant = False
                break
            const_coeffs[exponent] = coeff.value

        if all_coeffs_constant:
            self._const_coeffs = const_coeffs

    def coefficients(self) -> list[Expression]:
        coeffs = []
        for i in range(self.degree + 1):
            if i in self._coeffs:
                coeffs.append(self._coeffs[i])
            else:
                coeffs.append(Expression.wrap(0.0))

        return coeffs

    def has_constant_coefficients(self) -> bool:
        return self._const_coeffs is not None

    def eval(self, t: T) -> T:
        if self._const_coeffs is None:
            raise PolynomialError("Cannot evaluate without constant polynomial coefficients")

        results = 0
        for coeff in reversed(self._const_coeffs):
            results = coeff + t * results  # type: ignore
        return results  # type: ignore


class ExtractionError(Exception):
    ...


class PolynomialError(Exception):
    ...


def extract_polynomial(expr: Expression, wrt: Variable) -> Polynomial:
    """
    Requires:
     - Non-negative, integer powers on `wrt`
     - wrt is in Reals()
     - Output of expr is in Reals()
    """
    if wrt not in Reals():
        raise ExtractionError("'wrt' Variable must be an element of Reals.")
    if expr not in Reals():
        raise ExtractionError("'expr' must output Real values.")

    expr = expand(expr)
    poly_terms = _extract_chain_under(expr, Add)

    def zero():
        return Expression.wrap(0.0)

    coefficients: defaultdict[int, Expression] = defaultdict(zero)
    for term in poly_terms:
        coeff, exp = _coefficient_and_exponent(term, wrt)
        coefficients[exp] = coefficients[exp] + coeff

    return Polynomial(coefficients)


def _extract_chain_under(expr: Expression, operator: type[BinaryOperator]) -> list[Expression]:
    """Extract a chain from the expression under a specific operation."""
    if type(expr) is operator:
        return expr._extract_chain()
    else:
        return [expr]


def _extract_exponent(expr: Power, wrt: Variable) -> int:
    if expr.a != exact(wrt):
        raise ExtractionError(f"Polynomial exponent must be of form: {wrt}**n, got {expr}")
    if expr.b not in Ints() or not isinstance(expr.b, Constant):
        raise ExtractionError(f"Polynomial exponents must be constant integer, got {expr}")
    return expr.b.value


def _coefficient_and_exponent(term: Expression, wrt: Variable) -> tuple[Expression, int]:
    """Given an expression of the general form a * b * wrt**n * d, return (a * b * d, n)

    Requires:
     - wrt's power is a constant integer
     - wrt appears at most once in the multiplication chain
    """
    term_parts = _extract_chain_under(term, Multiply)

    # Only one part should have wrt (if any).
    exponent = 0
    coefficient = Expression.wrap(1)
    for part in term_parts:
        if not part.has_variable(wrt):
            coefficient = coefficient * part
            continue
        if exponent != 0:
            raise ExtractionError(f"Polynomial term has {wrt} in multiple parts ({term})")

        if type(part) is Power:
            exponent = _extract_exponent(part, wrt)
        elif part == exact(wrt):
            exponent = 1
        else:
            raise ExtractionError(f"Polynomial term must be either: Any * wrt**n or Any * wrt")

    return coefficient, exponent
