import io,\
       operator,\
       sys,\
       os,\
       mimetypes,\
       csv,\
       itertools,\
       json,\
       shutil,\
       glob,\
       pickle,\
       tarfile

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

NoneType = type(None)

def is_iter(obj):
    """Test wheter `obj` can be used in a `for` loop."""
    # Rank 0 tensors in PyTorch are not really iterable
    return isinstance(obj, (Iterable, Generator)) and getattr(obj, "ndim", 1)

def is_coll(obj):
    "Test wether `obj` can be used in a `for` loop"
    # Rank 0 tensors in PyTorch do not have working `len`
    return hasattr(obj, "__len__") and getattr(obj, "ndim", 1)

if __name__ == "__main__":
    print("hi")
