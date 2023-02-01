# Copyright

from __future__ import annotations

from collections import defaultdict
import logging
import types
from typing import Any

from .constant import Constant
from .set import Set
from .util import extract_return_type


__all__ = [
    "SetElement",
    "SetElementMapping",
    "SetElementError",
]


class SetElementError(Exception):
    ...


class SetElement(Constant):
    """TODO: Doc"""

    element_mappings: dict[type[Set], type[SetElementMapping]]

    _types_to_mappings: dict[type, list[type[SetElementMapping]]] = defaultdict(list)

    @classmethod
    def find(cls, value) -> SetElement | None:
        """Look for a SetElement with a successful mapping from the given value.

        New values, Sets and SetElements can be accomodated by defining a new SetElementMapping.
        """
        mappings = cls._types_to_mappings.get(type(value))
        if mappings is None:
            return None

        for mapping in mappings:
            element = mapping.to_symbol(value)
            if element is not None:
                return element

        return None

    def __init__(self, in_: Set):
        # PERFORMANCE: Consider the use of the ParameterSingletonMeta metaclass avoid
        # repeated instantiation of SetElements. Particularly in the binary_condense
        # methods (see condense_identity() for example).
        super().__init__(None, output_set=in_)

    def __init_subclass__(cls):
        cls.element_mappings = {}
        return super().__init_subclass__()

    @property
    def value(self):
        result = self._get_value(self.output_set)
        if result is not None:
            return result

        raise SetElementError(
            "No SetElementMapping found for SetElement and Set combo: "
            f"{type(self)} in {self.output_set}."
        )

    def _get_value(self, output_set) -> Any | None:
        mapping = self.element_mappings.get(type(output_set))
        if mapping is not None:
            result = mapping.from_symbol(output_set)
            if result is not None:
                return result

        # Failed for this output_set, try its supersets.
        for superset in output_set.supersets(strict=True):
            result = self._get_value(superset)
            if result is not None:
                return result

    @value.setter
    def value(self, v):
        if v is not None:
            raise TypeError("Cannot set value property of SetElement type")


class SetElementMapping:
    """TODO: Doc"""

    element: type[SetElement]
    set: Set
    static_value = None

    def __init_subclass__(cls):
        # Allow this mapping to be found in SetElement's class by looking up the Set type.
        set_type = type(cls.set)
        if set_type in cls.element.element_mappings:
            other = cls.element.element_mappings[set_type]
            if not cls.__module__.startswith(SetElementMapping.__module__):
                raise TypeError(f"Element mapping conflict within Symgrad: {other} and {cls}")

            logging.warning(f"Overriding existing SetElement mapping, {other} with {cls}")

        cls.element.element_mappings[set_type] = cls

        # Allow this mapping to be found in the SetElement superclass by
        # looking up the input value type.
        if cls.static_value is not None:
            value_type = type(cls.static_value)
        else:
            value_type, _ = extract_return_type(cls.from_symbol)
            if value_type is Any:
                raise NotImplementedError(
                    "SetElementMapping subclass much define either:\n"
                    " - static_value; or\n"
                    " - from_symbol() classmethod and annotate its return type without 'Any'."
                )

        map_list = SetElement._types_to_mappings[value_type]
        if cls not in map_list:
            map_list.insert(0, cls)

    @classmethod
    def to_symbol(cls, v: Any) -> SetElement | None:
        """Override this method if mapping from a concrete value requires more than an equality check."""
        if v == cls.from_symbol(cls.set):
            return cls.element(cls.set)

    @classmethod
    def from_symbol(cls, set: Set) -> Any:
        """Override this method if the output value depends on the Set parameters."""
        if cls.static_value is not None:
            return cls.static_value

        raise NotImplementedError(
            "SetElementMapping subclass much define either:\n"
            " - static_value; or\n"
            " - from_symbol() classmethod and annotate its return type without 'Any'."
        )
