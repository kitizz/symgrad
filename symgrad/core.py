# Copyright 2018, Christopher Ham.

import collections
import inspect
import logging

import numpy as np

logger = logging.Logger("Generators")


class Lookup:
    """A Lookup whose keys are expected to be a Sequence of class types.

    Requires:
     - All except the first element of a key must be a subclass of `base_class`.

    Ensures:
     - lookup will return values for keys whose elements after the first are
       subclasses of an added key. See tests below for examples.
    """

    def __init__(self, base_class):
        """
        Args:
            base_class (class): When traversing the class hierarchy, stop at this class.
        """
        self.base_class = base_class
        self.dirty = True
        self.sparse = collections.OrderedDict()
        self.dense = dict()
        self.source_map = dict()

    def add(self, key, value):
        names = tuple([k.__name__ for k in key])
        if names == ("Sutract", "Real", "Real"):
            print("Add key", names, value)
        if not all(isinstance(k, type) for k in key[1:]):
            raise ValueError("Keys must be a subclass of self.base_class")
        self.sparse[key] = value
        self.dirty = True

    def lookup(self, key, default=None):
        self._fill_dense()
        return self.dense.get(key, default)

    def _fill_dense(self):
        if not self.dirty:
            return

        self.dense = dict()
        keys = list(self.sparse.keys())
        dist = [self._dist_from_gen(key) for key in keys]
        # Process keys in descending order.
        for i in np.argsort(dist)[::-1]:
            key = keys[i]
            Lookup._fill_super_keys(self.dense, key, self.sparse[key], self.source_map)

        self.dirty = False

    def _dist_from_gen(self, key):
        dist = 0
        for cls in key[1:]:
            super_classes = cls.mro()
            for i in range(len(super_classes)):
                if super_classes[i] == self.base_class:
                    dist += i
                    break
            else:
                raise ValueError(
                    f'"{cls}" not derived from base_class, "{self.base_class}".'
                )
        return dist

    @staticmethod
    def _fill_super_keys(lookup, key, value, source_map, source_key=None):
        """First Generator is kept constant."""
        if key in lookup:
            return

        # print('Add key:', format_key(key))
        source_key = source_key or key
        source_map[key] = source_key
        lookup[key] = value
        for i in range(1, len(key)):
            new_key = list(key)
            for subcls in key[i].__subclasses__():
                new_key[i] = subcls
                Lookup._fill_super_keys(
                    lookup, tuple(new_key), value, source_map, source_key
                )


def format_key(grad_key):
    return "({})".format(", ".join(k.__name__ for k in grad_key))


def _header(func_name, func_sig):
    """Generate the header for a function."""
    sig = []
    for arg_name, param in func_sig.parameters.items():
        sig.append("{}: '{}'".format(arg_name, param.annotation.__name__))
        # sig.append('{}'.format(arg_name))
    return "def {}({}):".format(func_name, ", ".join(sig))


def symbolic_from_function(func):
    inputs = {}
    for arg_name, param in inspect.signature(func).parameters.items():
        cls = param.annotation
        inputs[arg_name] = cls(arg_name)
    return func(**inputs)


#
# ------------- TESTING below ---------------------
#


class TestLookup:
    class Base:
        ...

    class Foo(Base):
        ...

    class Bar(Base):
        ...

    class FooFoo(Foo):
        ...

    class NotBase:
        ...

    def test_basic(self):
        lookup = Lookup(self.Base)

        lookup.add((self.NotBase, self.Foo), "Foo")

        assert lookup.lookup((self.NotBase, self.Foo)) == "Foo"
        assert lookup.lookup((self.NotBase, self.FooFoo)) == "Foo"
        assert lookup.lookup((self.NotBase, self.Bar)) is None

        lookup.add((self.NotBase, self.Bar), "Bar")
        assert lookup.lookup((self.NotBase, self.Foo)) == "Foo"
        assert lookup.lookup((self.NotBase, self.FooFoo)) == "Foo"
        assert lookup.lookup((self.NotBase, self.Bar)) == "Bar"

    def test_idempotent(self):
        classes = [self.Foo, self.FooFoo]

        for before_order in (list, reversed):
            for after_order in (list, reversed):
                lookup = Lookup(self.Base)
                for cls in before_order(classes):
                    lookup.add((self.NotBase, cls), cls.__name__)

                for cls in classes:
                    assert lookup.lookup((self.NotBase, cls)) == cls.__name__

                for cls in after_order(classes):
                    lookup.add((self.NotBase, cls), cls.__name__)

                for cls in classes:
                    assert lookup.lookup((self.NotBase, cls)) == cls.__name__
