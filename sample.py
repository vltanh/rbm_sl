import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from rbm import TreeRBM
from contrastive_divergence import ContrastiveDivergence
from sl import StochasticLocalization

SCALE = 1

def eval(model):
    # Evaluate the model
    model.eval()

    # Generate the images
    test_dataset = datasets.MNIST('./data',
        train=False,
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((28 // SCALE, 28 // SCALE)),
            lambda x: 2 * (x > 0).float() - 1
        ])
    )
    vis_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    images = next(iter(vis_loader))[0]

    v = images.view(-1, 784 // (SCALE ** 2))
    v_gibbs = model(v)

    v = (v + 1) / 2
    v_gibbs = (v_gibbs + 1) / 2

    # # Show the real images
    # show_and_save(make_grid(v.view(64, 1, 28, 28).data), f'results/real_{epoch}')

    # Show the generated images
    show_and_save(make_grid(v_gibbs.view(64, 1, 28 // SCALE, 28 // SCALE).data), f'sample')

def show_and_save(img, file_name):
    """Show and save the image.
    Args:
        img (Tensor): The image.
        file_name (Str): The destination.
    """
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    plt.imshow(npimg, cmap='gray')
    plt.imsave(f, npimg)

n_v = 784 // (SCALE ** 2)
n_h = 784 // (SCALE ** 2)
n_epochs = 20
lr = 0.01

rbm = TreeRBM(n_v=n_v, n_h=n_h)
rbm.load_state_dict(torch.load('model.pt'))

model = ContrastiveDivergence(rbm, k=1)
# model = StochasticLocalization(rbm, L=10, delta=1.0)

eval(model)