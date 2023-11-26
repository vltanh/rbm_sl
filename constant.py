import torch

DEFAULT_TYPE = torch.float64
SEED = 0

DEBUG = False
CUDA = False

TREE_DATA = False
TREE_MODEL = True

ALGO = 'slfg'

assert not (not TREE_MODEL and ALGO == 'slfg'), \
    'Cannot use Factor Graph on non-tree model'
