from collections.abc import Callable
from dataclasses import dataclass, field
import logging
import numbers
import typing

from symgrad.cache import rule_cache
from symgrad.constant import Constant
from symgrad.expression import Expression
from symgrad.operators import Add, BinaryOperator, Multiply, Power, UnaryOperator, Neg, Inverse
from symgrad.manipulation.extract_chain import extract_sortable_chain, extract_associative_chain
from symgrad.sets import Reals
from symgrad.variable import Variable

__all__ = [
    "sort_chain",
]

logger = logging.getLogger(__name__)
logger.setLevel("WARNING")

Order: typing.TypeAlias = Callable[[Expression, Expression], bool]
OrderFactory: typing.TypeAlias = Callable[[Expression], Order]


def sort_chain(expr: Expression, order_factory: OrderFactory) -> Expression:
    """
    Ensures:
    - All sub Expressions are sorted from the bottom up.
    - Reordering occurs when extract_sortable_chains is successful on a sub Expression.
    
    Requires:
    - order compares two Expressions, and returns True if LHS should preceed RHS.
    """
    chain = extract_sortable_chain(expr)
    if len(chain) == 1:
        if isinstance(expr, UnaryOperator):
            return type(expr)(sort_chain(expr.a, order_factory))
        elif isinstance(expr, BinaryOperator):
            return type(expr)(sort_chain(expr.a, order_factory), sort_chain(expr.b, order_factory))
        else:
            return expr
    assert isinstance(expr, BinaryOperator)
    
    order = order_factory(expr)
    chain = [SortWrapper(sort_chain(term, order_factory), order) for term in chain]
    chain.sort()

    return type(expr).reduce((wrapper.expr for wrapper in chain))


@dataclass
class SortWrapper:
    expr: Expression
    order: Order
    
    def __lt__(self, other: "SortWrapper"):
        return self.order(self.expr, other.expr)