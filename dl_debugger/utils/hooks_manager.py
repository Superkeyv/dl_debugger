""" manage hook generate by torch.hook, and record need info from hook """

import pickle

from collections import defaultdict
from torch.utils.hooks import RemovableHandle
from typing import Any, Dict

from .config import DLSAN_DUMP_RAW_TENSOR, DLSAN_DUMP_TENSOR_DETAIL
from .data_struct import flatten_nested_tensor_feat, flatten_nested_tensor_cpu
from .dump_file import append_row_to_table, allocate_table, get_my_dump_folder
from .logging import logger


_HOOKS:Dict[str, Any] = {}

class HookManager:
    """
    Manager Hook Handler, and record data from hook.
    When hook collect finish in one cycle, store them to dump_file memory.


    Phase:      [ Fwd -> Bwd ] -> ... [ Fwd -> Bwd ] -> [ Optim.Step ]
                     |   /
    Data:         record 1      ...     record n

    Merge:        record

    Record:       append_row()

    Args:
        prefix: the file name for this table
    """

    def __init__(self, table_name:str) -> None:
        if table_name in _HOOKS:
            logger.warning('repeat instantiate hook_mgr %s', table_name)
        # remember this instance
        _HOOKS[table_name] = self

        self.name =  table_name
        self._table = allocate_table(table_name)
        self._hook_handles:dict[str, RemovableHandle] = {}
        self._hit_name = defaultdict(lambda: 0)

    def reg_handle(self, name, handle:RemovableHandle):
        """ add handle to manager, and remove exists one. """
        if name in self._hook_handles:
            _h = self._hook_handles.pop(name)
            _h.remove()
        self._hook_handles[name] = handle

    def is_registered(self, name):
        return name in self._hook_handles

    def add_to_table(self, name:str, value):
        # map name to value, and add them to correct position in the table
        self._hit_name[name] += 1

        row = {name: value}
        if DLSAN_DUMP_RAW_TENSOR:
            # save inline to save memory
            dump_folder = get_my_dump_folder()
            hit_count = self._hit_name[name]
            row = flatten_nested_tensor_cpu(row)
            with open(dump_folder / f'{hit_count}.{self.name}.{name}.pkl',
                      'wb') as f:
                pickle.dump(row, f)
            return
        elif DLSAN_DUMP_TENSOR_DETAIL:
            row = flatten_nested_tensor_feat(row, True)
        else:
            row = flatten_nested_tensor_feat(row, False)
        append_row_to_table(self._table, row)

    def get_handle_count(self) -> int:
        return len(self._hook_handles)

    def get_hit_count(self, name) -> int:
        return self._hit_name[name]

    def remove_hook(self, name):
        if not self.is_registered(name):
            return
        
        hd:RemovableHandle = self._hook_handles.pop(name)
        try:
            hd.remove()
        except:
            logger.exception('remove hook %s failed.', name)
        self._hit_name.pop(name)

    def reset(self):
        for name in self._hook_handles:
            self.remove_hook(name)

        self._hook_handles.clear()
        self._hit_name.clear()

    def clear(self):
        """ reset and clear self._table """
        self.reset()
        self._table.clear()
