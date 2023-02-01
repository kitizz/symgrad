from ..expression import Expression
from . import abstract_tree as tree
from ..operators import BinaryOperator, UnaryOperator
from ..constant import Constant
from ..variable import Variable
from ..set_element import SetElement


def abstract_tree_from_expression(expr: Expression) -> tree.TreeNode:
    """Parse an abstract TreeNode from a full Expression.

    This can be used for matching expressions. See Matcher()
    """
    match expr:
        case BinaryOperator() as op:
            a = abstract_tree_from_expression(op.a)
            b = abstract_tree_from_expression(op.b)
            return tree.Function(name=type(op).__name__, children=(a, b))

        case UnaryOperator() as op:
            a = abstract_tree_from_expression(op.a)
            return tree.Function(name=type(op).__name__, children=(a,))

        case Variable() as var:
            return tree.Term(name=var.name)

        case SetElement() as set_elem:
            return tree.Number(value=set_elem)

        case Constant() as const:
            try:
                value = float(const.value)
            except TypeError:
                raise TypeError(
                    "Unsupported Constant type. Expression matching "
                    "currently supports only float-convertible types. "
                    f"Got {type(const.value)}."
                )
            return tree.Number(value=value)

        case _:
            raise TypeError(f"match_pattern unable to handle type, {type(expr)}")
