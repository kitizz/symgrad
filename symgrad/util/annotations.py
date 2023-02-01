# Copyright

import inspect
import types
import typing
from typing import Any, Callable, TypeVar


__all__ = [
    "extract_input_types",
    "extract_return_type",
]


def extract_input_types(func: Callable[[Any], Any]) -> tuple:
    """Returns a tuple representing the Union of types allowed as inputs to func."""

    sig = inspect.signature(func)

    # Extract parameter types
    if len(sig.parameters) != 1:
        raise TypeError("Set mapping must define exactly one input.")

    param = next(iter(sig.parameters.values()))
    if param.annotation == inspect.Parameter.empty:
        raise TypeError(f"Annotation missing in function {func.__name__}")

    origin = typing.get_origin(param.annotation)
    if origin is typing.Union or origin is types.UnionType:
        param_types = typing.get_args(param.annotation)
    else:
        param_types = (param.annotation,)

    return param_types


T = TypeVar("T")


def extract_return_type(func: Callable[[Any], T | None]) -> tuple[type[T], bool]:
    """TODO: Doc"""

    sig = inspect.signature(func)

    # Extract return type
    if sig.return_annotation == inspect.Signature.empty:
        raise TypeError(f"Return annotation missing in function {func.__name__}")

    # return_types = get_types(sig.return_annotation)
    origin = typing.get_origin(sig.return_annotation)
    if origin is typing.Union or origin is types.UnionType:
        return_types = typing.get_args(sig.return_annotation)
        if (len(return_types) > 2) or (type(None) not in return_types):
            raise TypeError(
                "Return type cannot be a Union other than an equivalent of Optional. "
                f"Instead got {sig.return_annotation}"
            )

        # Get the non-None value.
        return_type = return_types[(return_types.index(type(None)) + 1) % 2]
        is_optional = True
    else:
        return_type = sig.return_annotation
        is_optional = False

    return return_type, is_optional
