import os

import torch
from torch.distributions import Categorical

from src.configurations import load_configurations
from src.energy_based import distribution, energy_rbm
from src.data import ArtificialDataset
from constant import DEFAULT_TYPE, TREE_DATA

torch.set_default_dtype(DEFAULT_TYPE)



