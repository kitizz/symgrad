from .exact import exact
from .sets import Reals
from .shared_context import no_auto_simplify
from .variable import Variable


def test_no_auto_simplify():
    x = Variable("x", Reals())

    assert x + x == exact(2 * x)

    with no_auto_simplify():
        assert x + x != exact(2 * x)
