import os
import time
from contextlib import contextmanager

import psutil

from symgrad.expression import Expression
from symgrad.manipulation.auto_simplify import auto_simplify
from symgrad.rules.knowledgebase import Knowledgebase
from symgrad.sets import Reals
from symgrad.sets.numbers import Ints
from symgrad.shared_context import define_rules, no_auto_simplify, run_test_with_knowledgebase
from symgrad.variable import Variable


@contextmanager
def foo():
    kbase = Knowledgebase()

    x = Variable("x", Reals())
    y = Variable("y", Reals())
    z = Variable("z", Reals())
    n = Variable("n", Ints())
    m = Variable("m", Ints())

    with run_test_with_knowledgebase(kbase):
        with define_rules():
            (x + y) in Reals()
            (x * y) in Reals()
            (x**y) in Reals()
            (-x) in Reals()

            (n + m) in Ints()
            (n * m) in Ints()
            (n**m) in Ints()
            (-n) in Ints()

            x + y == y + x
            x * y == y * x
            (x + y) + z == x + (y + z)
            (x + y) * z == x * z + y * z
            x * (y + z) == x * y + x * z  # TODO: Derive automatically...
            (x * y) ** n == x**n * y**n
            x ** (n + m) == (x**n) * (x**m)
            -(x + y) == (-x) + (-y)
            x + 0 == x
            x * 1 == x

            x - x == 0
            x * 0 == 0

            -(-x) == x
            -x == -1 * x
            (-x) * (-y) == x * y
            -(x * y) == x * (-y)
            -(x * y) == (-x) * y

            x**1 == x
            x**0 == 1
            x * x == x**2
            (x**n) * x == x ** (n + 1)
            x * (x**n) == x ** (n + 1)
            x + x == 2 * x
            (x**n) ** m == x ** (n * m)

        with no_auto_simplify():
            yield kbase
            kbase._stats.print()


if __name__ == "__main__":
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    c1 = Expression.wrap(3.5)
    c2 = Expression.wrap(-2)

    process = psutil.Process(os.getpid())
    print(f"Memory used before: {process.memory_info().rss // 1024 // 1024}mb")

    with foo():
        t = time.time()
        loops = 100
        for i in range(loops):
            auto_simplify((c1 * (-x)) * y)
            auto_simplify(c1 + (c2 + x))
            auto_simplify(0.5 * x**2 + 1.5 * x**2)
        dt = time.time() - t
        print(f"Memory used after: {process.memory_info().rss // 1024 // 1024}mb")
        print(f"Per iteration: {1000*dt/loops:.2f}ms")
