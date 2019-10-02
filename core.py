import os
import re
import types
import typing
import random
import inspect
import operator
import functools
import itertools

import pandas as pd
from copy import copy
from pathlib import Path
from test import is_iter
from typing import Generator, Iterator
from functools import partial
from operator import itemgetter
from contextlib import contextmanager
from IPython.core.debugger import set_trace

from imports import is_coll, NoneType

class FixSigMeta(type):
    "A metaclass that fixes the signautre on classes that override __new__"
    def __new__(cls, name, bases, dict):
        res = super().__new__(cls, name, bases, dict)
        # if inherit class has a modified copy signature
        if res.__init__ is not object.__init__:
            res.__signature__ = inspect.signature(res.__init__)
        return res

class PrePostInitMeta(FixSigMeta):
    "A metaclass that calls optional `__pre_init__` and `__init__` methods."
    def __call__(cls, *args, **kwargs):
        # create class and perform pre/post calls if exists
        res = cls.__new__(cls)
        if type(res) == cls:
            if hasattr(res, "__pre_init__"): res.__pre_init__(*args, **kwargs)
            res.__init__(*args, **kwargs)
            if hasattr(res, "__pre_init__"): res.__post_init__(*args, **kwargs)
        return res

class NewChkMeta(FixSigMeta):
    "Metaclass to avoid recreating object passed to constructor."
    def __call__(cls, x=None, *args, **kwargs):
        # if argument is same class with no arguments return the class
        if not args and not kwargs and x is not None and isinstance(x, cls):
            x._newchk = 1
            return x
        res = super().__call__(*((x,)+args), **kwargs)
        res._newchk = 0
        return res

class BypassNewMeta(FixSigMeta):
    """
    Metaclass: casts `x` to this class if it's of type `cls._bypass_type`,
    intializing with `_new_meta` if available"
    """
    def __call__(cls, x=None, *args, **kwargs):
        if hasattr(cls, "_new_meta"):
            x = cls._new_meta(x, *args, **kwargs)
        elif not isinstance(x, getattr(cls, "_bypass_type", object)) or len(args) or len(kwargs):
            x = super().__call__(*((x,)+args), **kwargs)
        if cls != x.__class__: x.__class__ = cls
        return x

def copy_func(func):
    "Copy a non-builtin function (NB `copy.copy` does not work for this)."
    # if not user defined function copy
    if not isinstance(func, types.FunctionType): return copy(func)
    fn = types.FunctionType(func.__code__, func.__globals__, func.__name__,
            func.__defaults__, func.__closure__)
    fn.__dict__.update(func.__dict__)
    return fn

def patch_to(cls, as_prop=False):
    "Decorators: add `f` to `cls`."
    if not isinstance(cls, (tuple, list)): cls = (cls,)
    def _inner(func):
        for c_ in cls:
            nf = copy_func(func)
            # copy all meta information
            for o in functools.WRAPPER_ASSIGNMENTS: setattr(nf, o, getattr(func, o))
            nf.__qualname__ = f"{c_.__name__}.{func.__name__}"
            setattr(c_, func.__name__, property(nf) if as_prop else nf)
        return func
    return _inner

def patch(func):
    "Decorator: add `func` to the first parameter's class (based on func's type annotations)."
    # neat method if first argument is class we can transer function to the class
    # however first argument becomes self
    cls = next(iter(func.__annotations__.values()))
    return patch_to(cls)(func)

def patch_property(func):
    """
    Decorator: add `func` as a property to the first parameters's class (based on
    f's type annotations).
    """
    cls = next(iter(func.__annotations__.values()))
    return patch_to(cls, as_prop=True)(func)

def _make_param(name, default=None):
    return inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default)

def use_kwargs(names, keep=False):
    "Decorator: replace `**kwargs` in signature with `names` params"
    def _func(func):
        sig = inspect.signature(func)
        sig_dict = dict(sig.parameters)
        kwargs = sig_dict.pop("kwargs")
        s2 = {name: _make_param(name) for name in names if name not in sig_dict}
        sig_dict.update(s2)
        if keep: sig_dict["kwargs"] = kwargs
        func.__signature__ = sig.replace(parameters=sig_dict.values())
        return func
    return _func

def delegates(to=None, keep=False):
    "Decorator: replace `**kwargs` in signature with params from `to`."
    def _func(func):
        # notation seems backward but what do I know
        if to is None: to_f, from_f = func.__base__.__init__, func.__init__
        else:          to_f, from_f = to, func

        sig = inspect.signature(from_f)
        sig_dict = dict(sig.parameters)
        kwargs = sig_dict.pop("kwargs")

        # grab all parameters from to_f ignoring overlap with from_f
        # and values that don't have default values
        s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
              if v.default != inspect.Parameter.empty and kwargs not in sig_dict}
        sig_dict.update(s2)

        if keep: sig_dict["kwargs"] = kwargs
        from_f.__signature__ = sig.replace(parameters=sig_dict.values())
        return func
    return _func

def funcs_kwargs(cls):
    "Replace methods in `self._methods` with those from `kwargs`."
    old_init = cls.__init__
    def _init(self, *args, **kwargs):
        for k in cls._methods:
            arg = kwargs.pop(k, None)
            if arg is not None:
                if isinstance(arg, types.MethodType):
                    # belongs to another class redo
                    arg = types.MethodType(arg.__func__, self)
                setattr(self, k, arg)
        old_init(self, *args, **kwargs)
    functools.update_wrapper(_init, old_init)
    cls.__init__ = use_kwargs(cls._methods)(_init)
    return cls

def method(func):
    "Mark `func` as a method"
    # `1` is a dummy instance since Py3 doesn't allow `None` any more
    return types.MethodType(func, 1)

@contextmanager
def working_directory(path):
    "Change working directory to `path` and return to previous on exit."
    prev_cwd = Path.cwd()
    os.chdir(path)
    try: yield
    finally: os.chdir(prev_cwd)

def add_docs(cls, cls_doc=None, **docs):
    """
    Copy values from `docs` to `cls` docstrings, and confirm all public
    methods are documented.
    """
    if cls_doc is not None: cls.__doc__ = cls_doc
    for k,v in docs.items():
        func = getattr(cls, k)
        if hasattr(func, "__func__"):
            # required for class methods
            func = func.__func__
        func.__doc__ = v
    # List of public callables without docstring
    nodoc = [c for n,c in vars(cls).items() if callable(c)
            and not n.startswith("_") and c.__doc__ is None]
    assert not nodoc, f"Missing docs: {nodoc}"
    assert cls.__doc__ is not None, f"Missing class docs: {cls}"

def docs(cls):
    "Decorator version of `add_docs`, using `_docs` dict."
    add_docs(cls, **cls._docs)
    return cls

def custom_dir(c, add:list):
    "Implement custom `__dir__`, adding `add` to `cls`"
    return dir(type(c)) + list(c.__dict__.keys()) + add

class GetAttr:
    """
    Inherit from this to have all attr accesses in `self._xtra`
    passed down to `self.default`
    """
    _default="default"
    @property
    def _xtra(self):
        # default if _xtra not defined
        return [o for o in dir(getattr(self, self._default))
                if not o.startswith("_")]

    def __getattr__(self, k):
        # how to handle attributes not defined by the class
        if k not in ("_xtra", self._default) and\
                (self._xtra is None or k in self._xtra):
            return getattr(getattr(self, self._default), k)
        raise AttributeError(k)

    def __dir__(self): return custom_dir(self, self._xtra)

    # for pickle and unpickle
    def __setstate__(self, data): self.__dict__.update(data)

def delegate_attr(self, k, to):
    """
    Use in `__getattr__` to delegate to attr `to` without inheriting
    from `Getattr`
    """
    # passing methods down
    if k.startswith("_") or k==to: raise AttributeError(k)
    try: return getattr(getattr(self, to), k)
    except AttributeError: raise AttributeError(k)

def _is_array(x):
    "if object is numpy or pandas array"
    return hasattr(x, "__array__") or hasattr(x, "iloc")

def _listify(obj):
    "Turn object to list if iterator, if string or array place in list."
    if obj is None: return []
    if isinstance(obj, list): return obj
    if isinstance(obj, str) or _is_array(obj): return [obj]
    if is_iter(obj): return list(obj)
    return [obj]

def coll_repr(col, max_n=10):
    "String repr of up to `max_n` items of (possibly lazy) collection `col`"
    string = f"(#{len(col)}) [" + ",".join(itertools.islice(map(str, col), max_n))\
            + ("..." if len(col) > 10 else "") + "]"
    return string

def mask2idxs(mask):
    "Convert bool mask or index list to index `L`"
    if isinstance(mask, slice): return mask
    mask = list(mask)
    if len(mask) == 0: return []
    if isinstance(mask[0], bool): return [i for i, m in enumerate(mask) if m]
    return [int(i) for i in mask]

listable_types = typing.Collection, typing.Generator, map, filter, zip

class CollBase:
    "Base class for composing a list of `items`"
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, k): return self.items[k]
    def __setitem__(self, i, v): self.items[list(k) if isinstance(k, CollBase) else k] = v
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self): return self.items.__repr__()
    def __iter__(self): return self.items.__iter__()

def cycle(obj):
    "Like `itertools.cycle` except creates list of `None`s if `obj` is empty"
    obj = _listify(obj)
    return itertools.cycle(obj) if obj is not None and len(obj) > 0 else itertools.cycle([None])

def zip_cycle(x, *args):
    """
    Like `itertools.zip_longest` but `cycle`s through elements of all
    but first argument
    """
    return zip(x, *map(cycle, args))

def is_indexer(idx):
    "Test wether `idx` will index a single item in a list"
    return isinstance(idx, int) or not getattr(idx, "ndim", 1)

class L(CollBase, GetAttr, metaclass=NewChkMeta):
    """
    Behaves like a list of `items` but can also index with
    list of indices or masks.
    """
    def __init__(self, items=None, *rest, use_list=False, match=None):
        if rest: items = (items,) + rest # ie. ([1,2], 3, 4)
        if items is None: items = [] # default to a list
        if (use_list is not None) or not _is_array(items):
            # mainly for strings, iterators, and arrays
            items = list(items) if use_list else _listify(items)
        if match is not None:
            # copy elements to match length, or assure items have len match
            if len(items) == 1: items = items*len(match)
            else: assert len(items) == len(match), "Match length mismatch"
        super().__init__(items)

    def _new(self, items, *args, **kwargs):
        return type(self)(items, *args, use_list=None, **kwargs)

    def __getitem__(self, idx):
        return self._get(idx) if is_indexer(idx) else L(self._get(idx), use_list=None)

    def _get(self, i):
        if is_indexer(i) or isinstance(i, slice):
            return getattr(self.items, "iloc", self.items)[i]
        i = mask2idxs(i)
        return (self.items.iloc[list(i)] if hasattr(self.items, 'iloc')
                else self.items.__array__()[(i,)] if hasattr(self.items, "__array__")
                else [self.items[i_] for i_ in i])

    def __setitem__(self, idx, obj):
        """Set `idx` (can be list of indices, or mask, or int) items to `obj`
        (which is brodcast if not iterable)"""
        idx = idx if isinstance(idx, L) else _listify(idx)
        if not is_iter(obj): obj = [obj]*len(idx)
        for i, obj_ in zip(idx, obj): self.items[i] = obj_

    @property
    def default(self):
        return self.items

    def __iter__(self):
        return iter(self.items.itertuples() if hasattr(self.items, "iloc")
               else self.items)

    def __contains__(self, b):
        return b in self.items

    def __invert__(self):
        return self._new(not i for i in self)

    def __eq__(self, b):
        return False if isinstance(b, (str, dict, set)) else all_equal(b, self)

    def __repr__(self):
        return repr(self.items) if _is_array(self.items) else coll_repr(self)

    def __mul__(a, b):
        return a._new(a.items*b)

    def __add__(a, b):
        return a._new(a.items + _listify(b))

    def __radd__(a, b):
        return a._new(b) + a

    def __addi__(a, b):
        a.items += list(b)
        return a

    def sorted(self, key=None, reverse=False):
       """
       New `L` sorted by `key`. If key is str use `attrgetter`. If key is
       int then use `itemgetter`.
       """
       if isinstance(key, str): k=lambda obj: getattr(obj, key, 0)
       elif isinstance(key, int): k=itemgetter(key)
       else: k=key
       return self._new(sorted(self.items, key=k, reverse=reverse))

    @classmethod
    def split(cls, s, sep=None, maxsplit=-1):
        return cls(s.split(sep, maxsplit))

    @classmethod
    def range(cls, a, b=None, step=None):
       """"
       Same as builtin `range`, but returns an `L`. Can pass a collection for
       `a`, to use `len(a)`
       """
       if is_coll(a): a = len(a)
       return (cls(range(a, b, step) if step is not None
               else range(a,b) if b is not None else range(a)))

    def map(self, func, *args, **kwargs):
        """
        Create new `L` with `func` applied to all `items`, passing `args` and
        `kwargs` to `func`."""
        f = (partial(func, *args, **kwargs) if callable(func)
                else func.format if isinstance(func, str) # print
                else func.__getitem__)                    # class
        return self._new(map(f, self))

    def unique(self):
        return L(dict.fromkeys(self).keys())

    def val2idx(self):
        return {v:k for k,v in enumerate(self)}

    def itemgot(self, idx):
        return self.map(itemgetter(idx))

    def attrgot(self, k, default=None):
        return self.map(lambda obj: getattr(obj, k, default))

    def cycle(self):
        return cycle(self)

    def filter(self, func, *args, **kwargs):
        return self._new(filter(partial(func, *args, **kwargs), self))

    def map_dict(self, func, *args, **kwargs):
        return {k:func(k, *args, **kwargs) for k in self}

    def starmap(self, func, *args, **kwargs):
        return self._new(itertools.starmap(partial(func, *args, **kwargs), self))

    def zip(self, cycled=False):
        return self._new((zip_cycle if cycled else zip)(*self))

    def zipwith(self, *rest, cycled=False):
        return self._new([self, *rest]).zip(cycled=cycled)

    def map_zip(self, func, cycled=False):
        return self.zip(cycled=cycled).starmap(func)

    def map_zipwith(self, func, *rest, cycled=False):
        return self.zipwith(*rest, cycled=cycled).starmap(func)

    def concat(self):
        return self._new(itertools.chain.from_iterable(self.map(L)))

    def shuffled(self):
        it = copy(self.items)
        random.shuffle(it)
        return self._new(it)

def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a

def get_class(nm, *fld_names, sup=None, doc=None, funcs=None, **flds):
    "Dynamically create a class, optionally inheriting from `sup`, containg `fld_name`"
    attrs = {}
    for f in fld_names: attrs[f] = None
    for f in L(funcs): attrs[f.__name__] = f
    for k,v in flds.items(): attrs[k] = v
    sup = ifnone(sup, ())
    if not isinstance(sup, tuple): sup=(sup,)

    def _init(self, *args, **kwargs):
        for i,v in enumerate(args): setattr(self, list(attrs.keys())[i], v)
        for k,v in kwargs.items(): setattr(self, k, v)

    def _repr(self):
        return "\n".join(f"{o}: {gegtattr(self, o)}" for o in set(dir(self))
                if not o.startswith("_") and not isinstance(getattr(self, o), types.MethodType))

    if not sup: attrs["__repr__"]= _repr
    attrs["__init__"] = _init
    res = type(nm, sup, attrs)
    if doc is not None: res.__doc__ = doc
    return res

def mk_class(nm, *fld_names, sup=None, doc=None, funcs=None, mod=None, **flds):
    "Create a class using `get_class` and add to the caller's module"
    if mod is None: mod = inspect.currentframe().f_back.f_locals
    res = get_class(nm, *fld_names, sup=sup, doc=doc, funcs=funcs, **flds)
    mod[nm] = res

def wrap_class(nm, *fld_names, sup=None, doc=None, funcs=None, **flds):
    "Decorator: makes functijon a method of a new class `nm` passing parameters to `mk_class`."
    def _inner(f):
        mk_class(nm, *fld_names, sup=sup, doc=doc, funcs=L(funcs)+f, mod=f.__globals__, **flds)
        return f
    return _inner

def noop(x=None, *args, **kwargs):
    "Do nothing."
    return x

def noops(self, x=None, *args, **kwargs):
    "Do nothing."
    return x

def store_attr(self, nms):
    "Store params named in comma-separated `nms` from calling context into attrs in `self`"
    mod = inspect.currentframe().f_back.f_locals
    for n in re.split(', *', nms): setattr(self, n, mod[n])

def attrdict(obj, *ks):
    "Dict from each `k` in `ks` to `getattr(o,k)`"
    return {k:getattr(obj, k) for k in ks}

def properties(cls, *ps):
    "Change attrs in `cls` with names in `ps` to properties"
    for p in ps: setattr(cls, p, property(getattr(cls, p)))

def tuplify(obj, use_list=False, match=None):
    "Make `obj` a tuple"
    return tuple(L(obj, use_list=use_list, match=match))

def replicate(item, match):
    "Create tuple of `item` copied `len(match)` times"
    return (item,)*len(match)

def uniqueify(x, sort=False, bidir=False, start=None):
    """Return the unique elements in `x`, optionally `sort`-ed,
    optionally return the reverse correspondance."""
    res = L(x).unique()
    if start is not None: res = start + res
    if sort: res.sort()
    if bidir: return res, res.val2idx()
    return res

def setify(obj):
    return obj if isinstance(obj, set) else set(L(obj))

def is_listy(x):
    return isinstance(x, (tuple, list, L, slice, Generator))

def range_of(x):
    "All indices of collection `x` (i.e. `list(range(len(x)))`)"
    return list(range(len(x)))

def groupby(x, key):
    """"Like `itertools.groupby` but doesn't need to be sorted,
    and isn't lazy"""
    res = {}
    for obj in x: res.setdefault(key(obj), []).append(obj)
    return res

def merge(*ds):
    "Merge all dictionaries in `ds`"
    return {k:v for d in ds if d is not None for k,v in d.items()}

def shufflish(x, pct=0.04):
    """Randomly relocate items of `x` up to `pct` of `len(x)` from
    their starting location"""
    n = len(x)
    return L(x[i] for i in sorted(range_of(x), key=lambda o: o+n*(1 + random.random()*pct)))

class IterLen:
    "Base class to add iteration to anything supporting `len` and `__getitem__`"
    def __iter__(self): return (self[i] for i in range_of(self))

class ReindexCollection(GetAttr, IterLen):
    """Reindexes collection `coll` with indices `idxs` and optional
    LRU cache of size `cache`"""
    _default = "coll"
    def __init__(self, coll, idxs=None, cache=None):
        self.coll, self.idxs, self.cache = coll, ifnone(idxs, L.range(coll)), cache
        def _get(self, i): return self.coll[i]
        self._get = types.MethodType(_get, self)
        if cache is not None: self._get = functools.lru_cache(maxsize=cache)(self._get)

    def __getitem__(self, i): return self._get(self.idxs[i])
    def __len__(self): return len(self.coll)
    def reindex(self, idxs): self.idxs = idxs
    def shuffle(self): random.shuffle(self.idxs)
    def cache_clear(self): self._get.cache_clear()

def _oper(op, a, b=None):
    return (lambda obj: op(obj, a)) if b is None else op(a, b)

def _mk_op(nm, mod=None):
    "Create an operator using `oper` and add to the caller's module"
    if mod is None: mod = inspect.currentframe().f_back.f_locals
    op = getattr(operator, nm)
    def _inner(a, b=None): return _oper(op, a, b)
    _inner.__name__ = _inner.__qualname__ = nm
    _inner.__doc__ = f"Same as `operator.{nm}`, or returns partial if 1 arg"
    mod[nm] = _inner

for op in "lt gt le ge eq ne add sub mul truediv".split(): _mk_op(op)

class _InfMeta(type):
    @property
    def count(self): return itertools.count()
    @property
    def zeros(self): return itertools.cycle([0])
    @property
    def ones(self): return itertools.cylce([1])
    @property
    def nones(self): return itertools.cylce([None])

class Inf(metaclass=_InfMeta):
    "Infinite lists"
    pass

def true(*args, **kwargs):
    "Predicate: always`True`"
    return True

def stop(e=StopIteration):
    "Raises exception `e` (by default `StopException`) even if in an expression"
    raise e

def gen(func, seq, cound=true):
    "Like `(cunf(o) for o in seq if cound(func(o))` but handles `StopIteration`"
    return iterools.takewhile(cond, map(func, seq))

def chunked(it, cs, drop_last=False):
    # it is list like or iterator and cs is size to chunk through
    if not isinstance(it, Iterator): it = iter(it)
    while True:
        res = list(itertools.islice(it, cs))
        if res and (len(res) == cs or not drop_last): yield res
        if len(res) < cs: return

def retain_type(new, old=None, typ=None):
    "Cast `new` to type of `old` if it's a superclass"
    # e.g. old is TensorImage, new is Tensor - if not subclass then do nothing
    if new is None: return new
    assert old is not None or typ is not None
    if typ is None:
        if not isinstance(old, type(new)): return new
        typ = old if isinstance(old, type) else type(old)
    # Do nothing the new type is already an instance of requested type (i.e. same type)
    return typ(new) if typ!=NoneType and not isinstance(new, typ) else new

def retain_types(new, old=None, typs=None):
    "Cast each item of `new` to type of matching item in `old` if it's a superclass"
    if not is_listy(new): return retain_type(new, old, typs)
    return type(new)(L(new, old, typs).map_zip(retain_type, cycled=True))

def show_title(o, ax=None, ctx=None, label=None, **kwargs):
    "Set title of `ax` to `o`, or print `o` if `ax` is `None`"
    ax = ifnone(ax, ctx)
    if ax is None: print(o)
    elif hasattr(ax, "set_title"): ax.set_title(o)
    elif isinstance(ax, pd.Series):
        while label in ax: label += "_"
        ax = ax.append(pd.Series({label: o}))
    return ax

class ShowTitle:
    "Base class that adds a simple `show`"
    _show_args = {"label": "text"}
    def show(self, ctx=None, **kwargs):
        return show_title(str(self), ctx=ctx, **merge(self._show_args, kwargs))

class Int(int, ShowTitle): pass
class Float(float, ShowTitle): pass
class Str(str, ShowTitle): pass

num_methods = """
    __add__ __sub__ __mul__ __matmul__ __truediv__ __floordiv__ __mod__ __divmod__ __pow__
    __lshift__ __rshift__ __and__ __xor__ __or__ __neg__ __pos__ __abs__
""".split()
rnum_methods = """
    __radd__ __rsub__ __rmul__ __rmatmul__ __rtruediv__ __rfloordiv__ __rmod__ __rdivmod__
    __rpow__ __rlshift__ __rrshift__ __rand__ __rxor__ __ror__
""".split()
inum_methods = """
    __iadd__ __isub__ __imul__ __imatmul__ __itruediv__
    __ifloordiv__ __imod__ __ipow__ __ilshift__ __irshift__ __iand__ __ixor__ __ior__
""".split()

class Tuple(tuple):
    "A `tuple` with elementwise ops and more friendly __init__ behavior"
    def __new__(cls, x=None, *rest):
        if x is None: x = ()
        if not isinstance(x, tuple):
            if len(rest): x = (x,)
            else:
                try: x = tuple(iter(x))
                except TypeError: x = (x,)
        return super().__new__(cls, x + rest if rest else x)

    def _op(self, op, *args):
        if not isinstance(self, Tuple): self = Tuple(self)
        return type(self)(map(op, self, *map(cycle, args)))

    def mul(self, *args):
        "`*` is already defiend `tuple` for replicating, so use `mul` instead"
        return Tuple._op(self, operator.mul, *args)

    def add(self, *args):
        "`+` is already defined in `tuple` for concat, so use `add` instaed"
        return Tuple._op(self, operator.add, *args)

def _get_op(op):
    if isinstance(op, str): op = getattr(operator, op)
    def _f(self, *args): return self._op(op, *args)
    return _f

for n in num_methods:
    if not hasattr(Tuple, n) and hasattr(operator, n): setattr(Tuple, n, _get_op(n))

for n in "eq ne lt le gt ge".split(): setattr(Tuple, n, _get_op(n))
setattr(Tuple, "__invert__", _get_op("__not__"))
setattr(Tuple, "max", _get_op(max))
setattr(Tuple, "min", _get_op(min))

class TupleTitled(Tuple, ShowTitle):
    "A `Tuple` with `show`"
    pass


def trace(func):
    "Add `set_trace` to an existing function `func`"
    def _inner(*args, **kwargs):
        set_trace()
        return func(*args, **kwargs)
    return _inner

def compose(*funcs, order=None):
    """Create a function that composes all functions in `funcs`, passing along
    remaining `*args` and `**kwargs` to all"""
    funcs = L(funcs)
    if order is not None: funcs = funcs.sorted(order)
    def _inner(x, *args, **kwargs):
        for f in L(funcs): x = f(x, *args, **kwargs)
        return x
    return _inner

def maps(*args, retain=noop):
    "Like `map`, except funcs are composed first"
    f = compose(*args[:-1])
    def _f(b): return retain(f(b), b)
    return map(_f, args[-1])

def partialler(f, *args, order=None, **kwargs):
    "Like `functools.partial` but also copies over docstring"
    fnew = partial(f, *args, **kwargs)
    fnew.__doc__ = f.__doc__
    if order is not None: fnew.order = order
    elif hasattr(f, "order"): fnew.order = f.order
    return fnew

def mapped(f, it):
    "map `f` over `it` unless it's not listy, in which case return  `f(it)`"
    return L(it).mapped(f) if is_listy(it) else f(it)
