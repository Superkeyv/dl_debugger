""" collect and dump all data automatic. """

import atexit
import datetime
import os
import pickle
import pyarrow as pa
import pytz
import torch

from collections import defaultdict
from pathlib import Path
from pyarrow import parquet as pq
from torch import Tensor
from typing import Dict, List

from .config import DLSAN_DUMP_DIR
from .logging import logger


_RANK = int(os.environ.get('RANK', -1))
_DUMP_DIR = None
logger.info(f' dlsan_dump output path: {DLSAN_DUMP_DIR}')


_start_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai')
                                    ).strftime("%Y%m%d-%H%M%S")


# define table dtype
table_dtype = Dict[str, List[float]]
_tables: Dict[str, table_dtype] = {}


def get_my_dump_folder(mid_name:callable=lambda: None):
    global _DUMP_DIR, _RANK
    if _DUMP_DIR is not None:
        return _DUMP_DIR
    
    _common_path = Path(DLSAN_DUMP_DIR)
    _my_folder = f'{_start_time}-rk{_RANK}'
    _mid_basename = mid_name()
    if _mid_basename:
        _my_folder += str(_mid_basename)

    _DUMP_DIR = _common_path / _my_folder
    if not os.path.exists(DLSAN_DUMP_DIR):
        os.makedirs(DLSAN_DUMP_DIR, exist_ok=True)
    
    return _DUMP_DIR


def allocate_table(name) -> table_dtype:
    if name not in _tables:
        _tables[name] = defaultdict(list)
    return _tables[name]


def append_row_to_table(table:table_dtype, row:Dict):
    """ if tabel record one row, the next row must be same struct.
    Args:
        row Dict[str, float|int|Tensor]: must be same struct
    """
    _sync_info_during_run()

    assert isinstance(row, Dict)
    for k, v in row.items():
        assert isinstance(v, (float, int, Tensor))
        table[k].append(v)


def _sync_info_during_run():
    global _RANK
    if _RANK <0 and torch.distributed.is_initialized():
        _RANK = torch.distributed.get_rank()


def _save_tables(tables:dict, filepath):
    
    def _dump_pkl_helper(obj, filename:str):
        with open(filename+'.pkl', 'wb') as f:
            pickle.dump(obj, f)

    os.makedirs(filepath, exist_ok=True)

    for name, table in tables.items():
        if len(table) == 0:
            continue
        
        file_prefix = f'{filepath}/{name}'
        is_tensor = False
        for v in table.values():
            is_tensor = len(v) > 0 and isinstance(v[0], torch.Tensor)
            break

        if is_tensor:
            logger.info(' - dump raw tensor to file %s', file_prefix)
            _dump_pkl_helper(table, file_prefix)
            continue

        try:
            _f = file_prefix + '.parquet'
            _t = pa.Table.from_pydict(table)
            pq.write_table(_t, _f)
            logger.info(' - save %s', filepath)
        except Exception as e:
            logger.exception(" - can't save table, dump to %s",
                             file_prefix + '.pkl',
                             exc_info=e)
            _dump_pkl_helper(table, file_prefix)


@atexit.register
def _save_at_exit():
    dump_folder = get_my_dump_folder()
    logger.info(' dump saved data to path: %s', dump_folder)
    _save_tables(_tables, dump_folder)
