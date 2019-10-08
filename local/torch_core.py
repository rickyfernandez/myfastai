from .core import *
from .imports import *
from .torch_imports import *
from fastprogress import progress_bar, master_bar

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

def set_seed(s):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)

def _fa_rebuild_tensor(cls, *args, **kwargs):
    return cls(torch._utils._rebuild_tensor_v2(*args, **kwargs))

def _fa_rebuild_qtensor(cls, *args, **kwargs):
    return cls(torch._utils._rebuild_qtensor(*args, **kwargs))

class TensorBase(Tensor, metaclass=BypassNewMeta):
    def _new_meta(self, *args, **kwargs): return tensor(self)

    def __reduce_ex__(self,proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (type(self), self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        if self.is_quantized: args = args + (self.q_scale(), self.q_zero_point())
        f = _fa_rebuild_qtensor if self.is_quantized else  _fa_rebuild_tensor
        return (f, args + (self.requires_grad, OrderedDict()))

def _patch_tb():
    def get_f(fn):
        def _f(self, *args, **kwargs):
            cls = self.__class__
            res = getattr(super(TensorBase, self), fn)(*args, **kwargs)
            return cls(res) if isinstance(res, Tensor) else res
        return _f

    t = tensor([1])
    skips = "__class__ __deepcopy__ __delattr__ __dir__ __doc__ __getattribute__ __hash__ __init__ \
        __init_subclass__ __new__ __reduce__ __reduce_ex__ __module__ __setstate__".split()

    for fn in dir(t):
        if fn in skips: continue
        f = getattr(t, fn)
        if isinstance(f, (MethodWrapperType, BuiltinFunctionType, BuiltinMethodType, MethodType, FunctionType)):
            setattr(TensorBase, fn, get_f(fn))

_patch_tb()

@patch
def tensored(self:L):
    "`mapped(tensor)`"
    return self.map(tensor)

@patch
def stack(self:L, dim=0):
    "Same as `torch.stack`"
    return torch.stack(list(self.tensored()), dim=dim)

@patch
def cat(self:L, dim=0):
    "Same as `torch.cat`"
    return torch.cat(list(self.tensored()), dim=dim)

def concat(*ls):
    "Concatenate tensors, arrays, lists, or tuples"
    if not len(ls): return []
    it = ls[0]
    if isinstance(it, torch.Tensor): res = torch.cat(ls)
    elif isinstance(it, np.ndarray): ers = np.concatenate(ls)
    else:
        res = itertools.chain.from_iterable(map(L, ls))
        if isinstance(it, (tuple, list)): res = type(it)(res)
        else: res = L(res)
    return retain_type(res, it)

class Chunks:
    "Slice and int indexing into a list of lists"
    def __init__(self, chunks, lens=None):
        self.chunks = chunks
        self.lens = L(map(len,self.chunks) if lens is None else lens)
        self.cumlens = np.cumsum(0+self.lens)
        self.totlen = self.cumlens[-1]

    def __getitem__(self,i):
        if isinstance(i,slice): return self.getslice(i)
        di,idx = self.doc_idx(i)
        return self.chunks[di][idx]

    def getslice(self, i):
        st_d,st_i = self.doc_idx(ifnone(i.start,0))
        en_d,en_i = self.doc_idx(ifnone(i.stop,self.totlen+1))
        res = [self.chunks[st_d][st_i:(en_i if st_d==en_d else sys.maxsize)]]
        for b in range(st_d+1,en_d): res.append(self.chunks[b])
        if st_d!=en_d and en_d<len(self.chunks): res.append(self.chunks[en_d][:en_i])
        return concat(*res)

    def doc_idx(self, i):
        if i<0: i=self.totlen+i # count from end
        docidx = np.searchsorted(self.cumlens, i+1)-1
        cl = self.cumlens[docidx]
        return docidx,i-cl

def one_param(m):
    "First parameter in `m`"
    return next(iter(m.parameters()))

def apply(func, x, *args, **kwargs):
    "Apply `func` recursively to `x`, passing on args"
    if is_listy(x): return type(x)([apply(func, o, *args, **kwargs) for o in x])
    if isinstance(x, dict): return {k: apply(func, v, *args, **kwargs) for k,v in x.items()}
    res = func(x, *args, **kwargs)
    return res if x is None else retain_type(res, x)

def to_detach(b, cpu=True):
    "Recursively detach lists of tensors in `b`; put them on the CPU if `cpu=True`."
    def _inner(x, cpu=True):
        if not isinstance(x, Tensor): return x
        x = x.detach()
        return x.cpu() if cpu else x
    return apply(_inner, b, cpu=cpu)

def to_half(b):
    "Recursively map lists of tensors in `b` to FP16."
    return apply(lambda x: x.half() if torch.is_floating_point(x) else x, b)

def to_float(b):
    "Recursively map lists of in tensors in `b` to float."
    return apply(lambda x: x.float() if torch.is_floating_point(x) else x, b)

# None: True if available; True: error fi not available; False: use CPU
defaults.use_cuda = None

def default_device(use_cuda=-1):
    "Return or set default device; `use_cuda`: None - CUDA if available; False - CPU"
    if use_cuda != -1: defaults.use_cuda = use_cuda
    use = defaults.use_cuda or (torch.cuda.is_available() and defaults.use_cuda is None)
    assert torch.cuda.is_available() or not use
    return torch.device(torch.cuda.current_device()) if use else torch.device("cpu")

def to_device(b, device=None):
    "Recursively put `b` on `device`."
    if device is None: device=default_device()
    def _inner(o): return o.to(device, non_blocking=True) if isinstance(o, Tensor) else o
    return apply(_inner, b)

def to_cpu(b):
    "Recursively map lists of tensors in `b ` to the cpu."
    return to_device(b, "cpu")

def to_np(x):
    "Convert a tensor to a numpy array."
    return apply(lambda o: o.data.cpu().numpy(), x)

def item_find(x, idx=0):
    "Recursively takes the `idx`-th element of `x`"
    if is_listy(x): return item_find(x[idx])
    if isinstance(x, dict):
        key = list(x.keys())[idx] if isinstance(idx, int) else idx
        return item_find(x[key])
    return x

def find_device(b):
    "Recusively search the device of `b`."
    return item_find(b).device

def find_bs(b):
    "Recursively search the batch size of `b`."
    return item_find(b).shape[0]

def np_func(f):
    "Convert a function taking and returning numpy arrays to one taking and returning tensors"
    def _inner(*args, **kwargs):
        nargs = [to_np(arg) if isinstance(arg, Tensor) else arg for arg in args]
        return tensor(f(*nargs, **kwargs))
    functools.update_wrapper(_inner, f)
    return _inner

class Module(nn.Module, metaclass=PrePostInitMeta):
    "Same as `nn.Module`, but no need for subclasses to call `super().__init__`."
    def __pre_init__(self, *args, **kwargs): super().__init__()
    def __init__(self): pass

from torch.nn.parallel import DistributedDataParallel

def get_model(model):
    "Return the model maybe wrapped inside `model`."
    return model.module if isinstance(model, (DistributedDataParallel, nn.DataParallel)) else model

def one_hot(x, c):
    "One-hot encode `x` with `c` classes."
    res = torch.zeros(c, dtype=torch.uint8)
    res[L(x)] = 1.
    return res

def one_hot_decode(x, vocab=None):
    return L(vocab[i] if vocab else i for i,x_ in enumerate(x) if x_==1)

def params(m):
    "Return all parameters of `m`"
    return [p for p in m.parameters()]

def trainable_params(m):
    "Return all trainable parameters of `m`"
    return [p for p in m.parameters() if p.requires_grad]

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

def bn_bias_params(m, with_bias=True):
    "Return all bias and BatchNorm parameters"
    if isinstance(m, bn_types): return L(m.parameters())
    res = L(m.children()).map(bn_bias_params, with_bias=with_bias).concat()
    if with_bias and hasattr(m, "bias"): res.append(m.bias)
    return res

def batch_to_samples(b, max_n=10):
    "`Transposes` a batch to (at most `max_n`) samples"
    if isinstance(b, Tensor): return list(b[:max_n])
    else:
        res = L(b).map(partial(batch_to_samples, max_n=max_n))
        return retain_types(res.zip(), [b])
