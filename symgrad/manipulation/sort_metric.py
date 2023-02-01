from dataclasses import dataclass, field
import logging

from symgrad.expression import Expression
from symgrad.operators import Add, BinaryOperator, Multiply, Power, UnaryOperator, Neg, Inverse
from symgrad.sets import Reals
from symgrad.variable import Variable


@dataclass
class SortMetric:
    variables: dict[str, float] = field(default_factory=dict)
    constant: float = 0.0

    def __add__(self, other) -> "SortMetric":
        if not isinstance(other, SortMetric):
            return NotImplemented
        variables = self.variables.copy()

        for name, other_val in other.variables.items():
            self_val = variables.get(name, 0)
            variables[name] = self_val + other_val
        return SortMetric(variables, self.constant + other.constant)

    def __mul__(self, other) -> "SortMetric":
        if not isinstance(other, SortMetric):
            return NotImplemented

        variables = self.variables.copy()
        for name, other_val in other.variables.items():
            self_val = variables.get(name, 0)
            variables[name] = self_val * other_val

        return SortMetric(variables, self.constant * other.constant)

    def __pow__(self, other) -> "SortMetric":
        if not isinstance(other, SortMetric):
            return NotImplemented

        # Ignore other.variables
        variables = {name: val**other.constant for name, val in self.variables.items()}
        return SortMetric(variables, self.constant**other.constant)

    def max(self, other: "SortMetric") -> "SortMetric":
        variables = self.variables.copy()
        for name, other_val in other.variables.items():
            self_val = variables.get(name, 0)
            variables[name] = max(self_val, other_val)
        return SortMetric(variables, max(self.constant, other.constant))


def _calculate_degree(expr: Expression) -> SortMetric:
    if expr.is_constant:
        return SortMetric(constant=0)

    if isinstance(expr, Variable):
        return SortMetric(variables={expr.name: 1})

    if isinstance(expr, BinaryOperator):
        a_degs = _calculate_degree(expr.a)
        b_degs = _calculate_degree(expr.b)

        if isinstance(expr, Power):
            if expr.b.is_constant and expr.b in Reals():
                return a_degs * SortMetric(constant=expr.b.eval())
            return a_degs

        if isinstance(expr, Multiply):
            return a_degs + b_degs

        if isinstance(expr, Add):
            return a_degs.max(b_degs)

        return SortMetric()

    if isinstance(expr, UnaryOperator):
        a_degs = _calculate_degree(expr.a)
        if isinstance(expr, Inverse):
            return a_degs * SortMetric(constant=-1)

        return SortMetric()

    raise NotImplementedError(f"Type not supported, {type(expr)}")


def _calculate_scale(expr: Expression) -> SortMetric:
    if expr.is_constant:
        return SortMetric(constant=expr.eval() if expr in Reals() else 1.0)

    if isinstance(expr, Variable):
        return SortMetric(variables={expr.name: 1})

    if isinstance(expr, BinaryOperator):
        a_degs = _calculate_scale(expr.a)
        b_degs = _calculate_scale(expr.b)

        if isinstance(expr, Power):
            return a_degs**b_degs

        if isinstance(expr, Multiply):
            return a_degs * b_degs

        if isinstance(expr, Add):
            return a_degs + b_degs

        return a_degs.max(b_degs)

    if isinstance(expr, UnaryOperator):
        a_degs = _calculate_scale(expr.a)
        if isinstance(expr, Neg):
            return a_degs * SortMetric(constant=-1)
        return a_degs

    raise NotImplementedError(f"Type not supported, {type(expr)}")
