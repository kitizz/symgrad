# Copyright

import inspect
from typing import Any, Self, TypeVar

from .expression import Expression

__all__ = [
    "SubclassDefinitionError",
    "ParameterSingletonMeta",
]

T = TypeVar("T")


class SubclassDefinitionError(TypeError):
    ...


class ParameterSingletonMeta(type):
    """This metaclass enforces instantiations of subclasses to be a singleton for each unique
    set of parameters. Parameters are defined in the body of the subclass definition.

    class Set(metaclass=ParameterSingletonMeta):
        ...

    class Array2d(Set):
        n: int
        m: int

    assert Array2d(4, 5) is Array2d(4, 5)
    assert Array2d(4, 5) is not Array2d(5, 4)
    """

    _instances = {}
    __set_signatures: dict[type, inspect.Signature] = {}

    def __call__(cls, *args, **kwargs):
        bound_args = cls.__bind_args(*args, **kwargs)
        instant_args = tuple(bound_args.values())
        key = (cls,) + instant_args

        if key not in cls._instances:
            instance = super(ParameterSingletonMeta, cls).__call__(*args, **kwargs)
            for name, value in bound_args.items():
                setattr(instance, name, value)
            instance._set_params = instant_args
            cls._instances[key] = instance
        return cls._instances[key]

    def __gen_signature(cls):
        """Ensure the _set_signature has an entry for this cls.

        Ensures:
         - All non-underscore class attributes are treated as Set parameters.
         - A Signature is created with parameters in the same order.

        Requires:
         - All parameters must be annotated.
         - The generated Signature can be bound with position-only args;
           i.e. only the last parameters may have default values.
        """
        if cls in cls.__set_signatures:
            return cls.__set_signatures[cls]

        params: dict[str, inspect.Parameter] = {}

        defaults = {}
        for name in cls.__dict__.keys():
            if not name.startswith("_"):
                # Need to use getattr() to make sure it properly binds staticmethod
                # so that it correctly returns True from callable()
                value = getattr(cls, name)
                if not callable(value):
                    defaults[name] = value

        for name, anno in inspect.get_annotations(cls).items():
            if not name.startswith("_"):
                params[name] = inspect.Parameter(
                    name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=anno,
                    default=defaults.get(name, inspect.Parameter.empty),
                )

        unannotated_defaults = tuple(set(defaults) - set(params))
        if unannotated_defaults:
            raise SubclassDefinitionError(
                f"Encountered unannotated parameters: {unannotated_defaults}"
            )

        sig = inspect.Signature(list(params.values()))
        cls.__set_signatures[cls] = sig

        return sig

    def __bind_args(cls, *args, **kwargs) -> dict[str, Any]:
        """Bind args and kwargs to the Set's parameters."""
        sig = cls.__gen_signature()
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return bound_args.arguments

    def __getattr__(cls, name):
        sig = cls.__gen_signature()
        param = sig.parameters.get(name, None)
        if param is None:
            raise AttributeError(f"type object '{cls.__name__}' has no attribute '{name}'")

        return Expression._spaghetti().Variable(name, param.annotation)

    def _param_names(cls) -> tuple[str, ...]:
        sig = cls.__gen_signature()
        return tuple(sig.parameters.keys())

    def prefixed_instance(cls: type[T], prefix: str) -> T:
        """Generate an instance of the class with parameters as variables with prefixed names."""
        # Unfortunately, at the time of coding, type checking wasn't quite able
        # to express metaclass types fully. So there are some ignores here so that
        # type checks in the broader library work nicely.
        sig: inspect.Signature = cls.__gen_signature()  # type: ignore
        prefixed_params = {}
        Variable = Expression._spaghetti().Variable
        for name in cls._param_names():  # type: ignore
            param = sig.parameters[name]
            prefixed_name = prefix + name
            prefixed_params[name] = Variable(prefixed_name, param.annotation)
        return cls(**prefixed_params)
