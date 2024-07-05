import copy

from torch import Tensor
from typing import Mapping, Tuple, Callable, List, Any, Dict

from .tensor_abstract import tensor_hash, tensor_fingerprint


ListTuple = (List, Tuple)


def nested_merge(a, b):
    """ merge a and b to c, not inplace op."""
    if a is None:
        return copy.deepcopy(b)
    if b is None:
        return copy.deepcopy(a)

    if isinstance(a, Mapping):
        assert isinstance(b, Mapping)
        rp = {}
        rp.update(a)
        for k in b:
            if k in a:
                rp[k] = nested_merge(a[k], b[k])
            else:
                rp[k] = copy.deepcopy(b[k])
        return rp

    if isinstance(a, ListTuple) or isinstance(b, ListTuple):
        rp = []
        if isinstance(a, ListTuple):
            rp.extend(a)
        else:
            rp.append(copy.deepcopy(a))

        if isinstance(b, ListTuple):
            rp.extend(b)
        else:
            rp.append(copy.deepcopy(b))
        return rp

    if isinstance(a, (int, float, str)) and isinstance(b, (int, float, str)):
        return [a, b]

    raise TypeError(f'not support type {type(a)} {type(b)}')


def check_nested_tensor_any(value, fn: Callable):
    """
    Args:
        value: check nested data which contain Tensor
        fn: callable function, perform check
            prtotype: fn(t: t.Tensor) -> bool

    Return:
        if fn return true once, the result is true
    """

    if isinstance(value, Tensor):
        return fn(value)

    if isinstance(value, ListTuple):
        flag = False
        for v in value:
            flag |= check_nested_tensor_any(v, fn)
        return flag

    if isinstance(value, Mapping):
        flag = False
        for _, v in value.items():
            flag |= check_nested_tensor_any(v, fn)
        return flag

    return False


def check_nested_tensor_all(value, fn: Callable):
    """
    Args:
        value: check nested data which contain Tensor
        fn: callable function, perform check
            prtotype: fn(t: t.Tensor) -> bool

    Return:
        if fn return false once, the result is false
    """

    if isinstance(value, Tensor):
        return fn(value)

    if isinstance(value, ListTuple):
        flag = True
        for v in value:
            flag &= check_nested_tensor_all(v, fn)
        return flag

    if isinstance(value, Mapping):
        flag = True
        for _, v in value.items():
            flag &= check_nested_tensor_all(v, fn)
        return flag

    return True


def flatten_nested_tensor_feat(obj, detail:bool=False) -> Dict[str, Any]:
    rp = {}
    def _flatten_helper(obj, name_list:list):
        prefix = '.'.join(name_list)
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                _flatten_helper(v, name_list+[str(k)])
        elif isinstance(obj, ListTuple):
            for i, v in enumerate(obj):
                _flatten_helper(v, name_list+[str(i)])
        elif isinstance(obj, Tensor):
            if detail:
                _flatten_helper(tensor_fingerprint(obj), name_list+['tensor'])
            else:
                rp[prefix+'.tensor'] = tensor_hash(obj)
        elif isinstance(obj, (float, int)) and not isinstance(obj, bool):
            rp[prefix+'.num'] = obj

    _flatten_helper(obj, [])
    return rp


def flatten_nested_tensor_cpu(obj) -> Dict[str, Tensor]:
    rp = {}
    def _flatten_helper(obj, name_list:list):
        prefix = '.'.join(name_list)
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                _flatten_helper(v, name_list+[str(k)])
        elif isinstance(obj, ListTuple):
            for i, v in enumerate(obj):
                _flatten_helper(v, name_list+[str(i)])
        elif isinstance(obj, Tensor):
            rp[prefix] = obj.detach().cpu()
        elif isinstance(obj, (float, int)) and not isinstance(obj, bool):
            rp[prefix] = obj

    _flatten_helper(obj, [])
    return rp
