import io
import re
import operator
import itertools
import numpy as np
from PIL import Image
from functools import partial
from collections import Counter
from contextlib import redirect_stdout
from collections.abc import Iterable, Generator

def test_fail(func, msg="", contains=""):
    """
    Fails with `msg` unless `func()` raises an exception and (optionally)
    has `contains` in `e.args`
    """
    try: func()
    except Exception as e:
        # if contains is defined check is subset
        # if e error message.
        assert not contains or contains in str(e)
        return
    assert False, f"Expected exception but none raised. {msg}"

def test(a, b, cmp, cname=None):
    """
    `assert` that `cmap(a,b)` display inputs and `cname or cmp.__name__`
    if it fails.
    """
    if cname is None: cname = cmp.__name__
    assert cmp(a, b), f"{cname}:\n{a}\n{b}"


def all_equal(a, b):
    """
    Compares whether `a` and `b` are the same length and have the same
    contents.
    """
    if not is_iter(b): return False
    return all(equals(a_, b_) for a_, b_ in itertools.zip_longest(a, b))

def one_is_instance(a, b, t):
    """True if a or b is instance of t."""
    return isinstance(a, t) or isinstance(b, t)

def equals(a, b):
    """Compares `a` and `b` for equality. Supports sublists, tensors and arrays too."""

    if one_is_instance(a, b, type): return a == b
    if hasattr(a, "__array_eq__"):  return a.__array_eq__(b)
    if hasattr(b, "__array_eq__"):  return b.__array_eq__(a)

    if one_is_instance(a, b, np.ndarray):
        cmp = np.array_equal
    elif one_is_instance(a, b, (str, dict, set)):
        cmp = operator.eq
    elif is_iter(a):
        cmp = all_equal
    else:
        cmp = operator.eq

    return cmp(a, b)

def nequals(a, b):
    """Compares `a` and `b` for `not equals`"""
    return not equals(a, b)

def test_eq(a, b):
    """`test` that `a==b`"""
    test(a, b, equals, "==")

def test_eq_type(a, b):
    """`test` that `a==b` and are same type"""
    test_eq(a, b)
    test_eq(type(a), type(b))
    if isinstance(a, (list, tuple)): test_eq(map(type, a), map(type, b))

def test_ne(a, b):
    """`test` that `a != b`"""
    test(a, b, nequals, "!=")

def is_close(a, b, eps=1e-5):
    """Is `a` within `eps` of `b`"""
    if hasattr(a, "__array__") or hasattr(b, "__array__"):
        return (abs(a-b)<eps).all()
    if isinstance(a, (Iterable, Generator)) or isinstance(b, (Iterable, Generator)):
        return is_close(np.array(a), np.array(b), eps=eps)
    return abs(a-b) < eps

def test_close(a, b, eps=1e-5):
    """`test` that `a` is within `eps` of `b`"""
    test(a, b, partial(is_close, eps=eps), "close")

def test_is(a,b):
    """`test` that `a is b`"""
    test(a, b, operator.is_, "is")

def test_shuffled(a, b):
    """`test` that `a` and `b` are shuffled versions of the same sequence of items"""
    test_ne(a, b)
    test_eq(Counter(a), Counter(b))

def test_stdout(func, exp, regex=False):
    """Test that `fucn` prints `exp` to stdout, optionally checking as `regex`"""
    s = io.StringIO()
    with redirect_stdout(s): func()
    if regex: assert re.search(exp, s.getvalue()) is not None
    else: test_eq(s.getvalue(), f"{exp}\n" if len(exp) > 0 else "")

def test_fig_exists(ax):
    """Test there is a figure displayed in `ax`"""
    assert ax and len(np.frombuffer(ax.figure.canvas.tostring_argb(), dtype=np.uint8))
