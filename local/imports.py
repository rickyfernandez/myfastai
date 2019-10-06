import io,\
       operator,\
       sys,\
       os,\
       re,\
       mimetypes,\
       csv,\
       itertools,\
       json,\
       shutil,\
       glob,\
       pickle,\
       tarfile,\
       collections,\
       hashlib,\
       random

from concurrent.futures import as_completed
from functools import partial, reduce
from itertools import starmap, dropwhile, takewhile, zip_longest
from copy import copy, deepcopy
from multiprocessing import Lock, Process, Queue, queues
from datetime import datetime
from contextlib import redirect_stdout, contextmanager
from collections.abc import Iterable, Iterator, Generator, Sequence
from typing import Union, Optional
from types import SimpleNamespace
from pathlib import Path
from collections import OrderedDict, defaultdict, Counter, namedtuple
from enum import Enum, IntEnum
from warnings import warn
from textwrap import TextWrapper
from operator import itemgetter, attrgetter, methodcaller
from urllib.request import urlopen

# External modules
#import requests, yaml, matplotlib.pyplot as plt, pandas as pd, scipy
import matplotlib.pyplot as plt, numpy as np, pandas as pd, scipy
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from IPython.core.debugger import set_trace

NoneType = type(None)
string_classes = (str, bytes)

def is_iter(obj):
    """Test wheter `obj` can be used in a `for` loop."""
    # Rank 0 tensors in PyTorch are not really iterable
    return isinstance(obj, (Iterable, Generator)) and getattr(obj, "ndim", 1)

def is_coll(obj):
    "Test wether `obj` can be used in a `for` loop"
    # Rank 0 tensors in PyTorch do not have working `len`
    return hasattr(obj, "__len__") and getattr(obj, "ndim", 1)

def all_equal(a, b):
    """Compares whether `a` and `b` are the same length and have the same
    contents."""
    if not is_iter(b): return False
    return all(equals(a_, b_) for a_, b_ in itertools.zip_longest(a, b))

def one_is_instance(a, b, t):
    "True if a or b is instance of t."
    return isinstance(a, t) or isinstance(b, t)

def equals(a, b):
    "Compares `a` and `b` for equality. Supports sublists, tensors and arrays too."
    if one_is_instance(a, b, type): return a == b
    if hasattr(a, "__array_eq__"):  return a.__array_eq__(b)
    if hasattr(b, "__array_eq__"):  return b.__array_eq__(a)

    if one_is_instance(a, b, np.ndarray): cmp = np.array_equal
    elif one_is_instance(a, b, (str, dict, set)): cmp = operator.eq
    elif is_iter(a): cmp = all_equal
    else: cmp = operator.eq

    return cmp(a, b)
