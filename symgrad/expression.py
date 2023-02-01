# Copyright 2018, Christopher Ham.

from __future__ import annotations

import dataclasses
import graphlib
import inspect
import logging
import typing
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Callable, NewType, TypeVar

from symgrad.shared_context import thread_local_context

if typing.TYPE_CHECKING:
    # These Spaghetti-related imports introduce dependency loops, but we want
    # type checking to keep the internal libraries safe.
    from .operators.binary_operator import BinaryOperator
    from .operators.unary_operator import UnaryOperator
    from .rules import Knowledgebase
    from .set import Set
    from .variable import Variable


class CodeDag:
    """Basically a DAG which maintains an association with underlying code wrt to graph nodes."""

    def __init__(self):
        self.sorter = graphlib.TopologicalSorter()
        self.code_lines = {}

    def add(self, assignee: str, code_line: str = "", deps: Sequence[str] = ()):
        if assignee in self.code_lines:
            if self.code_lines[assignee] != code_line:
                raise ValueError(
                    "assignee and code_line are expected to be consistent across calls"
                )
        else:
            self.code_lines[assignee] = code_line
            self.sorter.add(assignee, *deps)

    def sorted_code_lines(self):
        lines = (
            self.code_lines[node] for node in self.sorter.static_order() if node in self.code_lines
        )
        return [line for line in lines if line]

    def symbols(self):
        return [key for key, value in self.code_lines.items() if not value]


class ExpressionClassError(Exception):
    ...


class Expression:
    """TODO: Summary

    Consider renaming to Expression?
    """

    #
    # Public API
    #

    name: str
    # We do not have access to the Variable type yet. Consider using Protocol here if required.
    variables: dict[str, Variable]
    operator_count: int
    operand_count: int
    is_constant: bool
    constant_val: Any | None

    #: The type (or mathematical set) of the evaluated result of this expression.
    #     Some examples:
    #         x, y, A = Real("x"), Real("y"), Vec3("A")
    #         (x + y).output_set == Real
    #         (x * A).output_set == Vec3
    #
    #     Note: Strictly speaking, mathematically, this could be called output_set,
    #     but "type" seems less ambiguous.
    output_set: Set

    def __init__(self, name, output_set):
        self.name = name
        self.variables = dict()
        self.operator_count = 0
        self.operand_count = 0
        self.output_set = output_set
        self._expr_hash = None
        self.is_constant = False
        self.constant_val = None

    def __init_subclass__(cls):
        value = Expression.__subclasses.setdefault(cls.__name__, cls)
        if value != cls:
            raise ExpressionClassError(
                "Subclasses of Generator must have unique names. "
                f"Collision encountered: {cls.__module__}.{cls.__qualname__} and "
                f"{value.__module__}.{value.__qualname__}"
            )

    def generate(self, dag: CodeDag):
        dag.add(self.id_name())

    def has_variable(self, var: str | Variable) -> bool:
        """Check if a Variable or variable name appears in this expression."""
        if isinstance(var, str):
            return var in self.variables
        else:
            selfvar = self.variables.get(var.name)
            return (selfvar is not None) and (selfvar._hash_() == var._hash_())

    EvalKey = TypeVar("EvalKey", str, "Expression")

    def sub(self, subs: dict[Expression.EvalKey, Any] | None = None, **kw_subs) -> Expression:
        """Substitute any Variables in this Expression with constants or other Expressions.

        Ensures:
         - An Expression will be returned adhering to any variable substitutions.
        """
        return Expression.wrap(self.eval(subs, **kw_subs))

    def eval(self, subs: dict[Expression.EvalKey, Any] | None = None, **kw_subs) -> Any:
        """Evaluate this expression with variable substitutions.

        Ensures:
         - If all variables in this expression have non-variable substitutions,
           a fully evaluated result will be returned.
         - Otherwise an Expression will be returned adhering to any variable substitutions.
        """
        expr_subs: dict[str, Any] = {k: v for k, v in kw_subs.items()}
        if subs:
            for k, v in subs.items():
                if isinstance(k, Expression) and type(k).__name__ == "Variable":
                    k = k.name
                if not isinstance(k, str):
                    raise TypeError("Substitution keys must be Variable or str (Variable name).")
                expr_subs[k] = v

        return self._eval_(expr_subs)

    def id_name(self) -> str:
        """Returns a string that can be used as a valid Python identifier (for variable names)."""
        REPLACE = {
            ".": "_",
            ":": "_",
            "-": "neg",
            "+": "pls",
            "^": "pwr",
            "*": "times",
            "/": "div",
        }

        name = self.name
        for _from, _to in REPLACE.items():
            name = name.replace(_from, _to)

        if not name.isidentifier():
            name = "_" + name
        if not name.isidentifier():
            raise ValueError('Name, "{}", must also be a valid python variable name.'.format(name))
        return name

    Hash = NewType("Hash", int)

    def hash(self) -> Hash:
        """Return a hash of this Expression unique to the exact Variable names,
        Constants and structure.

        This camn be used to check if two Expressions are identical.
        """
        return typing.cast(Expression.Hash, self._hash_())

    def verbose(self):
        return str(self) + "{" + self.__class__.__name__ + "}"

    def to_string(self, *, format):
        """

        Possible formats:
         - debug: Shows underlying order of operations and keeps BinaryOperators pairs in brackets.
         - display: A more easily readable format for general display.
        """
        match format:
            case "debug":
                return self._str_debug(None)
            case "display":
                return self._str_display(None)
            case _:
                raise RuntimeError(f"Unknown format for Generator.to_string, '{format}'")

    @classmethod
    def wrap(cls, value) -> Expression:
        """Factory function to Expression-ify any value.

        Ensures:
         - Expression inputs are forwarded out.
         - If possible, any other values are wrapped as a Constant expression type;
           otherwise TypeError is thrown.
        """
        if isinstance(value, Expression):
            return value

        wrapped = cls.__subclasses["SetElement"].find(value)  # type: ignore
        if wrapped is None:
            wrapped = cls.__subclasses["Constant"](value)  # type: ignore

        assert isinstance(wrapped, Expression)
        return wrapped

    __spaghetti_partial: dict[str, Any] = {}
    __spaghetti: InnerSpaghetti | None = None

    @classmethod
    def _add_spaghetti(cls, name: str, value):
        spaghetti_field_names = {f.name for f in dataclasses.fields(InnerSpaghetti)}
        if name not in spaghetti_field_names:
            raise KeyError(f"InnerSpaghetti has no field called '{name}'")
        cls.__spaghetti_partial[name] = value

    @classmethod
    def _spaghetti(cls) -> InnerSpaghetti:
        """Access the internal InnerSpaghetti class and all its noodliness.

        Use this thoughtfully and as little as possible! The less Spaghetti,
        the easier this library is to maintain.

        Requires:
         - All InnerSpaghetti fields have been registered with _add_spaghetti.
        """
        if cls.__spaghetti is None:
            cls.__spaghetti = InnerSpaghetti(**cls.__spaghetti_partial)
        return cls.__spaghetti

    #
    # Built-in operator implementations
    #

    def _get_binary_op(self, name) -> type[BinaryOperator]:
        op = self._operators.get(name, None)
        if not op:
            raise ExpressionClassError(f"Missing operator: {name}")
        return op  # type: ignore

    def _get_unary_op(self, name) -> type[UnaryOperator]:
        op = self._operators.get(name, None)
        if not op:
            raise ExpressionClassError(f"Missing operator: {name}")
        return op  # type: ignore

    def __eq__(self, other):
        """TODO: Return an Equals() expression with the LHS and RHS

        Discussion:
        There were a few different behaviors that could reasonably have been
        implemented here, namely:
         1. Return if self and other are identical
         2. Return if self and other are equivalent
         3. Return an Expression object capturing the equality relationship.

        Option (1) seems too limited to me.
        But mainly, when considering '==' in the context of other comparison
        operators, then (1) and (2) both seem out of place. There's no
        equivalent (hah) of (1) for '<', '>', etc. And (2) may often be
        undecidable; ie. have no known solution.

        Option (3) won't try to induce any NP-hard logic, and allows for
        consistent behavior amongst the other comparitors.
        """
        if thread_local_context().defining_rules:
            kbase = self._spaghetti().the_knowledgebase()
            kbase.add(self, Expression.wrap(other))
            # TODO: Return the Equality expression once implemented.
            return None  # type: ignore
        else:
            return NotImplemented

    def __mul__(self, other):
        return self._get_binary_op("Multiply")(self, other)

    def __rmul__(self, other):
        return self._get_binary_op("Multiply")(other, self)

    def __truediv__(self, other):
        inv = self._get_unary_op("Inverse")
        return self._get_binary_op("Multiply")(self, inv(other))

    def __rtruediv__(self, other):
        inv = self._get_unary_op("Inverse")
        return self._get_binary_op("Multiply")(other, inv(self))

    def __add__(self, other):
        return self._get_binary_op("Add")(self, other)

    def __radd__(self, other):
        return self._get_binary_op("Add")(other, self)

    def __sub__(self, other):
        neg = self._get_unary_op("Neg")
        return self._get_binary_op("Add")(self, neg(other))

    def __rsub__(self, other):
        neg = self._get_unary_op("Neg")
        return self._get_binary_op("Add")(other, neg(self))

    def __pow__(self, other):
        return self._get_binary_op("Power")(self, other)

    def __neg__(self):
        return self._get_unary_op("Neg")(self)

    def __pos__(self):
        return self._get_unary_op("Pos")(self)

    def __getitem__(self, sl):
        return self._get_binary_op("Subscript")(self, sl)

    def sin(self):
        return self._get_unary_op("Sin")(self)

    def cos(self):
        return self._get_unary_op("Cos")(self)

    def sqrt(self):
        return self._get_unary_op("Sqrt")(self)

    @property
    def T(self):
        return self._get_unary_op("Transpose")(self)

    #
    # Interface for subclasses to implement
    #
    @abstractmethod
    def _eval_(self, substitutions: dict[str, Any]):
        raise NotImplementedError(
            f"Subclasses of Expression, {type(self)}, must implement _eval_()"
        )

    @abstractmethod
    def _hash_(self) -> int:
        """An expression _hash allows quick checks on whether two expressions are identical.

        Implementations are encouraged to cache the hash result.
        """
        raise NotImplementedError(f"Subclass of Expression, {type(self)}, must implement _hash.")

    #
    # Private details.
    #

    # All classes that subclass Generator.
    # Note that some method implementations expect certain classes to be available
    # here. It's a bit hacky and increases the spaghetti factor of this library.
    # Each time a dependency is added to __subclasses, the implementation should
    # call it out and justify how the API ergonomics make it worthwhile.
    #
    # Below maintains a list of dependencies:
    #  - make_variable() requires the Variable class
    __subclasses: dict[str, type[Expression]] = {}

    # All class that subclass Operator.
    _operators: dict[str, type] = {}

    def _str_debug(self, parent: Expression | None):
        return self.name

    def _str_display(self, parent: Expression | None):
        return self._str_debug(parent)

    def __str__(self):
        return self._str_display(parent=None)

    def __repr__(self):
        return str(self)


@dataclasses.dataclass(frozen=True)
class InnerSpaghetti:
    """This class helps make explicit the choice to allow some spaghetti
    dependencies in the library. Whenever they're invoked, the hope is that
    this classes makes it obvious and easy to follow trail in both directions
    whenever needed in development.

    On multiple occassions, some lower-level parts of the library needs access
    to a higher-level part. This occurs particularly often around the Operator
    classes that rely on higher-level routines to do things internally. But then
    those routines usually depend on the BinaryOperator interfaces in some way.

    For example, BinaryOperators use the Knowledgebase to check the expected
    output Set for operating on the two input Sets.
    """

    #: See rules/knowledgebase.py
    the_knowledgebase: Callable[[], Knowledgebase]

    #: See variable.py
    Variable: type[Variable]


def _header(func_name: str, func_sig: inspect.Signature):
    """Generate the header for a function.

    Roughly like:
        "def {func_name}(stringify(func_sig)):"
    """

    def render_arg(item):
        name, parameter = item
        return f'{name}: "{parameter.annotation.__name__}"'

    arg_str = ", ".join(map(render_arg, func_sig.parameters.items()))
    return f"def {func_name}({arg_str}):"


def render_single(gen: Expression, func_name: str, func_sig: inspect.Signature) -> str:
    """
    Returns:
        Rendered eval-able code string defining a function that execute
        the underlying operations.
    """
    dag = CodeDag()
    gen.generate(dag)
    # Remove duplicate work.
    lines = dag.sorted_code_lines()
    lines = [_header(func_name, func_sig)] + lines + ["return {}".format(gen.name)]
    return "\n    ".join(lines)


def merge_unique_ordered(list_a: list, list_b: list):
    """Returns a new list with duplicates removed from the two input lists.

    It also maintains the relative partial ordering of the input lists.

    Raises:
        - ValueError when duplicates don't maintain partial ordering.
          Eg: [1, 2, 3] vs [2, 1, 3]
    """
    dups: list[tuple[int, int]] = [(-1, -1)]
    for i, val_a in enumerate(list_a):
        for j, val_b in enumerate(list_b):
            if val_a == val_b:
                dups.append((i, j))
    dups.append((len(list_a), len(list_b)))

    print("\n".join(map(str, dups)))

    for ind in range(1, len(dups)):
        if dups[ind - 1][1] > dups[ind][1]:
            print(list_a)
            print(list_b)
            raise ValueError("Can't handle cross-over duplicates without proper DAG!")

    output = []
    for ind in range(1, len(dups)):
        from_a, from_b = dups[ind - 1]
        to_a, to_b = dups[ind]
        output.extend(list_a[from_a + 1 : to_a])
        output.extend(list_b[from_b + 1 : to_b])
        if to_a < len(list_a):
            output.append(list_a[to_a])

    return output


def render_multi(gens: Sequence[Expression], func_name: str, func_sig: inspect.Signature) -> str:
    """Render the code to output an array of Generator results.

    Makes some ordered assumptions about the underlying code generation to
    remove redundant lines.
    """
    dag = CodeDag()
    for gen in gens:
        gen.generate(dag)

    return_lines = ["return ["]
    return_lines += (f"    {gen.id_name()}," for gen in gens)
    return_lines.append("]")
    lines = [_header(func_name, func_sig)] + dag.sorted_code_lines() + return_lines
    return "\n    ".join(lines)
