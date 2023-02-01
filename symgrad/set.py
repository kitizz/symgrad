# Copyright, Christopher Ham, 2022

from __future__ import annotations
from abc import ABC
import abc

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import graphlib
import inspect
import logging
import typing
import types
from typing import Any, Callable

from symgrad.shared_context import thread_local_context

from .expression import Expression
from .parameter_singleton_meta import ParameterSingletonMeta
from .util import extract_input_types, extract_return_type

if typing.TYPE_CHECKING:
    from .variable import Variable


# TODO: Delete?
class NonContainmentError(TypeError):
    def __init__(self, a: type[Set], b: type[Set]):
        self.a = a
        self.b = b
        super().__init__(
            f"Sets {a} and {b} could not be related by containment "
            "(ie. neither is a subset of the other)."
        )


class SetMappingError(Exception):
    ...


class Set(metaclass=ParameterSingletonMeta):
    """The base class used for defining any mathematical set to be used by this library.

    A new Set can be defined by inheriting from this Set class. Look at other Sets
    defined in this directory for examples. Note that a Set may have no parameters
    like Real() or Int(). Sets can also take parameters (not unlike C++ templates)
    that are sometimes needed to identify a specific Set; for example
    Matrix(Real(), 3, 3) denotes a 3x3 matrix of real numbers (R^(3x3)).

    Very little behavior is defined in the Set class itself. As in the realm of
    mathematics, most structure comes from rules that define how specific operators
    can act on various sets. See `UnaryRule` and `BinaryRule` for how this can be
    achieved.

    Rules for defining a new Set class:
     - Set parameters are defined by class attributes (order is maintained).
     - All parameters must be annotated.
     - The annotated parameters must be able to form a function Signature that
       can be bound with position-only args; i.e. only the tail parameters may
       have default values.

    Further, if you'd like to enable static type analysis for the Set, there are two
    relatively easy options:
     1. Use the @typing.type_check_only decorator on an __init__ method that does
        nothing whose signature matches the equivalent signature of the Set parameters.
     2. Define a parallel .pyi file that likewise defines an __init__.

    At the time of writing, the there was no obvious way to autogenerate typing info.
    """

    #
    # Public API
    #
    def params(self) -> dict[str, Any]:
        """An ordered dict mapping this Set's parameter names to this instance's values."""
        return {name: value for name, value in zip(type(self)._param_names(), self._set_params)}

    @classmethod
    def find(cls, value) -> Set | None:
        """Find the smallest subset that has been registered for the given value.

        Ensures:
         - If value is a Set, value is returned.
         - If value is a Set class that can be instantiated with zero args,
           its instantiation will be returned.
         - If value is a class/type, the most inclusive registered Set type that can be
           instantiated with zero args will be returned.
         - For all other values, try the set mappings registered for type(value) inputs
           in order of least to most inclusive Set outputs. When their are ambiguities
           in Set containment (eg One, Zero) the last defined mapping is tried first.
        """
        if isinstance(value, Set):
            return value

        if isinstance(value, type) and issubclass(value, Set):
            try:
                return value()
            except TypeError:
                pass

        # Note that the set_mapping lists should be ordered by their return Set types.
        # The smallest (sub-est) set to largest (super-est) set.
        if isinstance(value, type):
            logging.warning(f"Looking up class/type: {value}")
            # Can't run the actual mapping with a class/type, so just choose the largest
            # superset we can instantiate with no args.
            for set_mapping in reversed(cls._set_mappings[value]):
                try:
                    return set_mapping.return_type()
                except TypeError:
                    pass
        else:
            for set_mapping in cls._set_mappings[type(value)]:
                maybe_set = set_mapping.func(value)
                if maybe_set:
                    return maybe_set

        return None

    @classmethod
    def like(cls, *args, **kwargs) -> Set:
        """TODO: Doc"""
        instance = cls(*args, **kwargs)
        assert hasattr(instance, "_set_params")
        return instance

    def eval_parameters(self, var_map: dict[Variable, Any]) -> Set:
        """Create a new Set by evaluating this Set's parameters with var_map.

        Examples:
          Foo(a, b).eval_parameters({a: 1, b: 2}) == Foo(1, 2)
          Foo(a, b).eval_parameters({a: 1}) == Foo(1, b)
          Foo(a, b).eval_parameters({}) == Foo(a, b)
        """
        if not var_map:
            return self

        params = self.params()
        new_params = {
            name: p.eval(var_map) if isinstance(p, Expression) else p for name, p in params.items()
        }
        return type(self)(**new_params)

    def _iterate_supersets(self, strict=False):
        """Yield pairs of instatiated supersets and each one's "parent" subset."""
        if not strict:
            yield self, self
        to_process = [self]
        while to_process:
            next_sets = []
            for s in to_process:
                superset_map = s._supersets()
                params = s.params()
                for superset_cls, arg_symbols in superset_map.items():
                    # TODO: Change _supersets behavior to reuse eval_parameters()
                    args = tuple(
                        v.eval(params) if isinstance(v, Expression) else v for v in arg_symbols
                    )
                    superset = superset_cls(*args)
                    yield superset, s
                    next_sets.append(superset)
            to_process = next_sets

    def superset_of(self, superclass) -> Set | None:
        """Return the instantiated superclass based on this set's parameters.
        Returns None if superclass is not an actual superset class of this Set.
        """
        for superset, subset in self._iterate_supersets():
            if type(superset) is superclass:
                return superset

    def supersets(self, *, strict=False) -> list[Set]:
        """Generate an ordered list of Sets that are supersets of this Set.

        Ensures:
         - First element of returned list is always self.
         - Given two elements a, b in the returned list, b will always come after a
           if b is a superset of a. Otherwise the relative order is undefined.
        """
        sorter = graphlib.TopologicalSorter()
        sorter.add(self)

        for superset, subset in self._iterate_supersets(strict=True):
            sorter.add(superset, subset)

        order = list(sorter.static_order())
        if strict:
            order.remove(self)
        return order

    @classmethod
    def is_superset_class(cls, other: type[Set], strict=False) -> bool:
        """Returns whether a Set type appears in the supersets for this Set."""
        to_process = [cls]
        while to_process:
            next_sets = []
            for s in to_process:
                if s is other and not (strict and s is cls):
                    return True
                next_sets.extend(s._supersets().keys())
            to_process = next_sets
        return False

    def superset_distance(self, other: Set) -> int:
        for distance, superset in enumerate(self.supersets()):
            if other is superset:
                return distance
        return -1

    #
    # Override these in Set implementations!
    #

    @classmethod
    def validate(cls, value) -> bool:
        raise NotImplementedError("validate() method must be implemented for Set subclass")

    @classmethod
    def sample(cls, size=None, seed=None) -> Sequence | None:
        """Override this method if you'd like to validate Rules defined for operations on this Set."""
        return None

    @classmethod
    def _supersets(cls) -> dict[type[Set], tuple]:
        """Override this to return a mapping of uninstatiated superset classes
        to parametric instantiation args. Example:

        class MySet:
            size: int

            def _supersets(cls):
                return {SomeSuperset: (cls.size,)}

        """
        return {}

    def __contains__(self, value) -> bool:
        """TODO: Doc"""
        if value is self:
            return True

        if isinstance(value, Set):
            # TODO: Make this work for Variable set parameters.
            return any(v is self for v in value.supersets())

        if thread_local_context().defining_rules:
            # Watch out! This won't work if implementing Set overrides __contains__
            kbase = Expression._spaghetti().the_knowledgebase()
            if binary_key := _binary_op_key(value):
                op, a_set, b_set = binary_key
                kbase.add_binary_set_rule(op, a_set, b_set, self)
                return True
            elif unary_key := _unary_op_key(value):
                op, a_set = unary_key
                kbase.add_unary_set_rule(op, a_set, self)
                return True

        if isinstance(value, Expression):
            return value.output_set in self

        return Expression.wrap(value).output_set in self

    #
    # Internal Implementation
    #

    _set_params: tuple

    @dataclass
    class SetMapping:
        #: The order that this object was added to the set_mappings list.
        order: int

        #: The return Set type of func.
        return_type: type[Set]

        #: Whether the return type in optional.
        return_optional: bool

        #: One of the allowed input types for func.
        input_type: type

        #: The actual mapping function.
        func: Callable[[Any], Set | None]

        def __lt__(self, other: Set.SetMapping):
            """Aim to have subsets first. When ambiguous have the most recently defined mapping first."""

            def set_class_lt(a: type[Set], b: type[Set]):
                """Order related Sets such that subsets come before their super sets."""
                return a.is_superset_class(b) and not b.is_superset_class(a)

            return set_class_lt(self.return_type, other.return_type) or (
                not set_class_lt(other.return_type, self.return_type) and self.order > other.order
            )

    # For any input type, keep track of functions that can map that type to a Set object.
    _set_mappings: dict[type, list[SetMapping]] = defaultdict(list)

    @classmethod
    def _register_mapping(cls, func: Callable[[Any], Set | None]):
        """Set @register_set_mappings() for detailed documentation."""
        input_types = extract_input_types(func)
        return_type, is_optional = extract_return_type(func)

        if not issubclass(return_type, Set):
            raise SetMappingError("Set mapping must return a Set (or subclass) object or None")

        for in_type in input_types:
            mapping_list = cls._set_mappings[in_type]
            set_mapping = cls.SetMapping(len(mapping_list), return_type, is_optional, in_type, func)

            if any(v.return_type == return_type for v in mapping_list):
                raise SetMappingError(
                    f"A Set mapping already exists that can map '{in_type}' to '{return_type}'."
                )

            mapping_list.append(set_mapping)
            # Aim to have subsets earlier. When ambiguous have the most recently defined mapping first.
            mapping_list.sort()

    def __init__(self, *args, **kwargs):
        ...

    def __init_subclass__(cls):
        ...

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash((type(self).__name__, self._set_params))

    def __repr__(self):
        return self._str_debug()

    def _str_debug(self) -> str:
        cls = type(self)
        args = [str(arg) for arg in self._set_params]
        return f"{cls.__name__}[{', '.join(args)}]"


class Null(Set):
    """A special set that is a subset of all other sets."""

    def __contains__(self, item):
        return item is Null()


def _get_set(obj, field: str) -> Set | None:
    value = getattr(obj, field, None)
    if not value:
        return
    set_ = getattr(value, "output_set", None)
    if not isinstance(set_, Set):
        return None
    return set_


def _binary_op_key(value) -> tuple[type, Set, Set] | None:
    if not (a_set := _get_set(value, "a")):
        return None
    if not (b_set := _get_set(value, "b")):
        return None

    return type(value), a_set, b_set


def _unary_op_key(value) -> tuple[type, Set] | None:
    if not (a_set := _get_set(value, "a")):
        return None
    if hasattr(value, "b"):
        return None

    return type(value), a_set


def register_set_mapping(func: Callable[[Any], Set | None]):
    """Register a function that maps a value onto its Symbolic Set object.

    Rules for registering Set mappings:
     - Function must take exactly one argument.
     - Input argument and return type must be annotated.
     - Return type must be a Set type or optional Set type.
       Valid: Int, Int | None, Union[Int, None], Optional[Int]
     - No two mappings can map the same input type to the same output type.
     - When two mappings can take the same input type, the return type of one
       must be a strict subset of the other return type.
    """
    Set._register_mapping(func)
