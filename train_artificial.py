import random
import datetime

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configurations import load_configurations
from contrastive_divergence import ContrastiveDivergence
from brute_force import marginal_rbm_bf
from generate_data import load_data
from rbm import TreeRBM, RBM
from sl import StochasticLocalization

from constant import DEFAULT_TYPE, SEED, TREE_MODEL, TREE_DATA, ALGO

torch.set_default_dtype(DEFAULT_TYPE)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def total_variation_distance(p, q):
    return 0.5 * torch.sum(torch.abs(p - q))


def train(model, train_loader, n_epochs=20, lr=0.01, true_dist=None, writer=None):
    """Train a RBM model.
    Args:
        model: The model.
        train_loader (DataLoader): The data loader.
        n_epochs (int, optional): The number of epochs. Defaults to 20.
        lr (Float, optional): The learning rate. Defaults to 0.01.
    Returns:
        The trained model.
    """
    # optimizer
    train_op = optim.Adam(model.parameters(), lr)

    # test the RBM model initialization
    learned_W = model.rbm.W.detach().type(DEFAULT_TYPE)
    y = torch.zeros(1, n)
    pred_dist = marginal_rbm_bf(learned_W, y)[0]

    tv_dist = total_variation_distance(pred_dist, true_dist).detach().item()
    writer.add_scalar('train/TV', tv_dist, 0)

    # train the RBM model
    model.train()

    loss_ = []
    pbar = tqdm(range(1, n_epochs + 1))
    pbar.set_description_str('Epoch %2d | TV=%.4f' % (0, tv_dist))
    for epoch in pbar:
        for true_v in tqdm(train_loader, leave=False):
            true_v = true_v.type(DEFAULT_TYPE)
            fantasy_v = model(true_v)
            loss = model.rbm.free_energy(true_v) \
                - model.rbm.free_energy(fantasy_v)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            if TREE_MODEL:
                model.rbm.mask_grad()
            train_op.step()

        with torch.no_grad():
            learned_W = model.rbm.W.detach().type(DEFAULT_TYPE)
            pred_dist = marginal_rbm_bf(learned_W, y)[0]
            tv_dist = \
                total_variation_distance(pred_dist, true_dist).detach().item()
            writer.add_scalar('train/TV', tv_dist, epoch)
            pbar.set_description_str('Epoch %2d | TV=%.4f' % (epoch, tv_dist))

    return model


timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
outdir = f'logs/{timestamp}'
if TREE_DATA:
    outdir += '_treedata'
else:
    outdir += '_fulldata'
if TREE_MODEL:
    outdir += '_treemodel'
else:
    outdir += '_fullmodel'
outdir += f'_{ALGO}'
writer = SummaryWriter(outdir)

n_v, n_h = 4, 2
n = n_v + n_h

N = 1000000

set_seed(SEED + 0)
if TREE_MODEL:
    rbm = TreeRBM(n_v, n_h)
else:
    rbm = RBM(n_v, n_h)

if ALGO == 'cd':
    model = ContrastiveDivergence(rbm, k=1)
elif ALGO == 'sl':
    model = StochasticLocalization(rbm, L=10, delta=1.)
else:
    raise NotImplementedError

set_seed(SEED + 1)
dataset = load_data(n_v, n_h, N)
loader = DataLoader(dataset, batch_size=1000, shuffle=True)

true_W = dataset.true_W.type(DEFAULT_TYPE)
y = torch.zeros(1, n).type(DEFAULT_TYPE)
true_dist = marginal_rbm_bf(true_W, y)[0]
print('True:', true_dist)

set_seed(SEED + 2)
train(model, loader, n_epochs=30, lr=0.00005,
      true_dist=true_dist, writer=writer)
torch.save(model.state_dict(), f'{outdir}/model.pt')
writer.close()
