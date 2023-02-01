from . import Real


def test_str_add():
    x = Real("x")
    y = Real("y")

    assert str(x + x) == "x + x"
    assert str(y + x + 10) == "(y + x) + 10"


def test_str_subtract():
    x = Real("x")
    y = Real("y")

    assert str(x - x) == "x - x"
    assert str(x - x - y) == "(x - x) - y"


def test_str_multiply():
    x = Real("x")
    y = Real("y")

    assert str(x * x) == "x * x"
    assert str(10 * x * y) == "(10 * x) * y"


def test_str_divide():
    x = Real("x")
    y = Real("y")

    assert str(x / x) == "x / x"
    assert str(10 / x / y) == "(10 / x) / y"


def test_str_polynomial():
    x = Real("x")

    assert str(4 * x**2 + 2 * x + 3) == "(4 * x**2 + 2 * x) + 3"
    assert str(4 * x**2 - 2 * x + 3) == "(4 * x**2 - 2 * x) + 3"
    assert str(-4 * x**2 - 2 * x - 3) == "(-4 * x**2 - 2 * x) - 3"


def test_to_string_display_add():
    x = Real("x")
    y = Real("y")

    def to_str(gen):
        return gen.to_string(format="display")

    assert to_str(x + x) == "x + x"
    assert to_str(x + y + 10) == "x + y + 10"
    assert to_str(x - x) == "x - x"
    assert to_str(-x - y - 10) == "-(x + y + 10)"
    assert to_str(x - y - 10) == "x - (y + 10)"


def test_to_string_display_multiply():
    x = Real("x")
    y = Real("y")

    def to_str(gen):
        return gen.to_string(format="display")

    assert to_str(x * x) == "x * x"
    assert to_str(10 * x * y) == "10 * x * y"
    assert to_str(x / x) == "x / x"
    assert to_str((1 / x) * (1 / y) * (1 / 10)) == "0.1 / (x * y)"


def test_to_string_display_polynomial():
    x = Real("x")

    def to_str(gen):
        return gen.to_string(format="display")

    assert to_str(x**2 + x + 1) == "x**2 + x + 1"
    assert to_str(x + x**2 + 1) == "x + x**2 + 1"
    assert to_str(x + 3 * x**2 - 2) == "x + 3 * x**2 - 2"


def test_extract_chain_add():
    x = Real("x")
    y = x**2 + 2 * x + 10

    assert y._extract_chain() == [x**2, 2 * x, 10]


def test_extract_chain_pow():
    x = Real("x")
    y = x**2

    assert y._extract_chain() == [x, 2]
