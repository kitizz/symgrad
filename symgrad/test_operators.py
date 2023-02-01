from . import Real

import logging

# y = (2 - (3 * x)) * (1 + (-4 * x) + (x ** 2))

# y = (2 - x + x**2)
# print([str(el) for el in y._extract_chain()])

# print(y)
# print()
# print(expand(y))

# y = cox_de_boor(x, [0, 1, 2, 3, 4], segment=1)

# y = ((((-0.5 * x) + (0.8333333333333333 * (x ** 2))) + (0.6666666666666666 * ((-1 + x) ** 2))) - ((((0.3333333333333333 * x) * ((0.5 * (x ** 2)) + (-0.5 * x))) + (0.16666666666666666 * (x ** 3))) + ((0.16666666666666666 * x) * ((-1 + x) ** 2))))
# y = (0.3333333333333333 * x) * ((0.5 * (x ** 2)) + (-0.5 * x))
# y = (0.3333333333333333 * x) * (-0.5 * x)
# print(y)
# print()
# print(expand(y))


def test_to_string_group():
    x = Real("x")
    assert str(x) == "x"


def test_associative_chain_add():
    x = Real("x")

    y = 2 - x + x**2
    add_chain, sub_chain = extract_associative_chain(y)
    assert len(add_chain) == 2
    assert len(sub_chain) == 1

    y = 2 + x - x**2
    add_chain, sub_chain = extract_associative_chain(y)
    assert len(add_chain) == 2
    assert len(sub_chain) == 1

    y = 2 + x - x**2 - 0.1 * x**4
    add_chain, sub_chain = extract_associative_chain(y)
    assert len(add_chain) == 2
    assert len(sub_chain) == 2

    y = 2 + x - (x**2 - 0.1 * x**4)
    add_chain, sub_chain = extract_associative_chain(y)
    assert len(add_chain) == 3
    assert len(sub_chain) == 1


def test_associative_chain_multiply():
    x = Real("x")

    y = 2 * x / x**2
    mul_chain, div_chain = extract_associative_chain(y)
    assert len(mul_chain) == 2
    assert len(div_chain) == 1

    y = x / x**2 / 4
    mul_chain, div_chain = extract_associative_chain(y)
    assert len(mul_chain) == 1
    assert len(div_chain) == 2

    y = x / (x**2 / 4)
    mul_chain, div_chain = extract_associative_chain(y)
    assert len(mul_chain) == 2
    assert len(div_chain) == 1


def test_simplify_add_easy():
    # TODO:...
    x = Real("x")
    y = Real("y")
    z = x + y - x

    assert extract_associative_chain(z) == ([x, y], [x])

    z_simp = simplify(z)
    assert z_simp != z
    assert z_simp == y


def test_simplify_add_easy():
    # TODO:...
    x = Real("x")
    y = Real("y")
    z = 2 * x + y - x

    assert extract_associative_chain(z) == ([2 * x, y], [x])

    z_simp = simplify(z)
    assert z_simp != z
    assert z_simp == x + y


def test_simplify_multiply():
    # TODO:...
    ...
