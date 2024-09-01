# copyright ZJX

import os

DLSAN_LOG_LEVEL = os.environ.get('DLSAN_LOG_LEVEL', 'info')
DLSAN_DUMP_DIR = os.environ.get('DLSAN_DUMP', None)
DLSAN_DUMP_TENSOR_DETAIL = os.environ.get('DLSAN_TENSOR_DETAIL', None)
DLSAN_DUMP_RAW_TENSOR = os.environ.get('DLSAN_DUMP_RAW_TENSOR', None)


if DLSAN_DUMP_RAW_TENSOR:
    assert DLSAN_DUMP_DIR, "Please specify env: DLSAN_DUMP_DIR for dump data."
