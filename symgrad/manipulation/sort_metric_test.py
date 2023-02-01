import logging

from symgrad.expression import Expression
from symgrad.manipulation.sort_metric import SortMetric, _calculate_degree, _calculate_scale
from symgrad.sets import Ints, Reals
from symgrad.variable import Variable


def test_degree(basic_knowledgebase):
    a = Variable("a", Reals())
    b = Variable("b", Reals())
    c = Variable("c", Ints())
    d = Variable("d", Reals())

    assert _calculate_degree(Expression.wrap(5)) == SortMetric()
    assert _calculate_degree(a) == SortMetric({"a": 1})
    assert _calculate_degree(a * b) == SortMetric({"a": 1, "b": 1})
    assert _calculate_degree(a**2) == SortMetric({"a": 2})