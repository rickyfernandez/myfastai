NoneType = type(None)

def is_coll(obj):
    "Test wether `obj` can be used in a `for` loop"
    # Rank 0 tensors in PyTorch do not have working `len`
    return hasattr(obj, "__len__") and getattr(obj, "ndim", 1)
