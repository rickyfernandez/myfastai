import os
import numpy as np
import pandas as pd

from imports import *
from torch_imports import *
from core import\
        patch,\
        is_iter

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("DEFAULT_GPU") or 0))

@patch
def __array__eq__(self:Tensor, b):
    return torch.equal(self, b) if self.dim() else self==b

def tensor(x, *rest, **kwargs):
    """Like `torch.as_tensor`, but handle lists too, and can pass multiple
    vector elemnts directly."""
    if len(rest): x = (x,) + rest
    # Pytorch bug in dataloader using num_workers>0
    if isinstance(x, (tuple, list)) and len(x)==0: return tensor(0)
    res = (torch.tensor(x, **kwargs) if isinstance(x, (tuple, list))
          else as_tensor(x.values, **kwargs) if isinstance(x, (pd.Series, pd.DataFrame))
          else as_tensor(x, **kwargs) if hasattr(x, "__array__") or is_iter(x)
          else None)
    if res is None:
        res = as_tensor(np.array(x), **kwargs)
        if res.dtype is torch.float64: return res.float()
    if res.dtype is torch.int32:
        warn("Tensor is int32: upgrading to int64; for better performance use int64 input")
        return res.long()
    return res



