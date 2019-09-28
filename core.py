import os
import types
import typing
import inspect
import functools
import itertools
from copy import copy
from pathlib import Path
from test import is_iter
from typeguard import typechecked
from contextlib import contextmanager


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

class BaseObj(metaclass=PrePostInitMeta):
    """Base class that provides `PrePostInitMeta` metaclass to subclasses."""
    pass

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

def runtime_check(func):
    # make sure arguments are correct type
    return typechecked(always=True)(func)

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
    @property
    def _xtra(self):
        # default if _xtra not defined
        return [o for o in dir(self.default) if not o.startswith("_")]

    def __getattr__(self, k):
        # how to handle attributes not defined by the class
        if k not in ("_xtra", "default") and\
                (self._xtra is None or k in self._xtra):
            return getattr(self.default, k)
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

def _is_array(x): return hasattr(x, "__array__") or hasattr(x, "iloc")

def _listify(obj):
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
        if items is None: items = []
        if (use_list is not None) or not _is_array(items):
            items = list(items) if use_list else _listify(items)
        if match is not None:
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
        return repr(self.items) if _is_arra(self.items) else coll_repr(self)

    def __mul__(a,b):
        return a.__new(a.items*b)

    def __add__(a, b):
        return a.__new(a.iterms + _listify(b))

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
    def range(cls, a, b=None, step=None):
       """"
       Same as builtin `range`, but returns an `L`. Can pass a collection for
       `a`, to use `len(a)`
       """
       if is_coll(a): a= len(a)
       return (cls(range(a, b, step) if step is not None 
               else range(a,b) if b is not None else range(a)))

    def unique(self):
        return L(dict.fromkeys(self).keys())
        
    def val2idx(self):
        return {v:k for k,v in enumerate(self)}

    def itemgot(self, idx):
        return self.mapped(itemgetter(idx))

    def attrgot(self, k, default=None):
        return self.mapped(lambda obj: getattr(obj, k, default))

    def cycle(self):
        return cycle(self)

    def filtered(self, func, *args, **kwargs):
        return self._new(filter(partial(func, *args, **kwargs), self))

    def mapped(self, func, *args, **kwargs):
        return self._new(map(partial(func, *args, **kwargs), self))

    def mapped_dict(self, func, *args, **kwargs):
        return {k:func(k, *args, **kwargs) for k in slef}

    def starmapped(self, func, *args, **kwargs):
        return self._new(itertools.starmap(partial(func, *args, **kwargs), self))

    def zipped(self, cycled=False):
        return self._new([self, *rest]).zipped(cycle=cyclded)

    def zippedwith(self, func, cycled=False):
        return self.zipped(cycled=cycled).starmapped(func)

    def mapped_zip(self, func, cycled=False):
        return self.zipped(cycled=cycled).starmapped(fnc)

    def mapped_zipwith(self, func, cycled=False):
        return self.zipwith(*rest, cycled=cycled).starmapped(func)

    def concat(self):
        return self._new(itertools.chain.from_iterable(self.mapped(L)))
    
    def shuffled(self):
        it = copy(self.items)
        random.shuffle(it)
        return self._new(it)

    """itemgetter, is_coll, random"""
