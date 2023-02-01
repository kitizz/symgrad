import random
from symgrad.expression import Expression
from symgrad.operators import BinaryOperator


def random_reduce(op: type[BinaryOperator], values: list, shuffle=False) -> Expression:
    """Reduce a chain of expressions with op, with random associations. Optionally shuffle."""
    values = list(values)
    if shuffle:
        random.shuffle(values)
    while len(values) > 1:
        index = random.randint(0, len(values) - 2)
        smoosh = op(values[index], values[index + 1])
        values[index] = smoosh
        del values[index + 1]

    return values[0]
