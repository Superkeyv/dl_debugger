""" Init log env. """

import logging
import torch

from .config import DLSAN_LOG_LEVEL


level_relations = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}
__log_level = level_relations.get(DLSAN_LOG_LEVEL)

class _CustomFormatter(logging.Formatter):
    def format(self, record):
        """ append rank info to message."""
        rank = -1 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
        record.msg = f'[Rank {rank}] {record.msg}'
        return super().format(record)


logger = logging.getLogger('Deep Learning Sanitizer')
logger.propagate = False
logger.setLevel(__log_level)

# configure stream handler
_sh = logging.StreamHandler()
_sh.setLevel(__log_level)
_sh.setFormatter(_CustomFormatter(
    '[%(asctime)s] [%(levelname)s] [%(filename)s]:%(lineno)d:%(funcName)s] %(message)s'
))

logger.addHandler(_sh)
