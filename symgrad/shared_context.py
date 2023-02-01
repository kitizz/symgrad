import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass

__local_data = threading.local()

__all__ = [
    "thread_local_context",
    "no_auto_simplify",
    "define_rules",
]


@dataclass
class LocalContext:
    # TODO: Doc
    auto_simplify: bool = True

    knowledgebase = None

    defining_rules: bool = False

    use_cache: bool = True


def thread_local_context() -> LocalContext:
    """TODO: Doc"""
    if not hasattr(__local_data, "context"):
        __local_data.context = LocalContext()
    return __local_data.context


@contextmanager
def no_auto_simplify():
    prev = thread_local_context().auto_simplify
    try:
        thread_local_context().auto_simplify = False
        yield
    finally:
        thread_local_context().auto_simplify = prev


@contextmanager
def run_test_with_knowledgebase(knowledgebase):
    """To be used only in the testing environment"""
    # if "PYTEST_CURRENT_TEST" not in os.environ:
    #     raise RuntimeError("test_with_knowledgebase context is only to be used during tests")

    prev = thread_local_context().knowledgebase
    try:
        thread_local_context().knowledgebase = knowledgebase  # type: ignore
        yield
    finally:
        thread_local_context().knowledgebase = prev


@contextmanager
def define_rules():
    prev = thread_local_context().defining_rules

    try:
        thread_local_context().defining_rules = True
        # Nest the no_auto_simplify
        with no_auto_simplify():
            yield
    finally:
        thread_local_context().defining_rules = prev


@contextmanager
def no_cache():
    prev = thread_local_context().use_cache

    try:
        thread_local_context().use_cache = False
        yield
    finally:
        thread_local_context().use_cache = prev
