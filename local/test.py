__all__ = ["test_fail", "test", "nequals", "test_eq", "test_eq_type", "test_ne", "is_close",
           "test_close", "test_is", "test_shuffled", "test_stdout", "TEST_IMAGE",
           "test_fig_exists"]

from .imports import *

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
    "Test that `func` prints `exp` to stdout, optionally checking as `regex`"
    s = io.StringIO()
    with redirect_stdout(s): func()
    if regex: assert re.search(exp, s.getvalue()) is not None
    else: test_eq(s.getvalue(), f"{exp}\n" if len(exp) > 0 else "")

TEST_IMAGE = "images/puppy.jpg"

def test_fig_exists(ax):
    "Test there is a figure displayed in `ax`"
    assert ax and len(np.frombuffer(ax.figure.canvas.tostring_argb(), dtype=np.uint8))

if __name__ == "__main__":
    print("hi")
