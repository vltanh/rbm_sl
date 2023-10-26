import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from rbm import TreeRBM
from contrastive_divergence import ContrastiveDivergence
from sl import StochasticLocalization

SCALE = 1

def eval(model, epoch):
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
    show_and_save(make_grid(v_gibbs.view(64, 1, 28 // SCALE, 28 // SCALE).data), f'results/fake_{epoch}')

def train(model, train_loader, n_epochs=20, lr=0.01):
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

    # train the RBM model
    model.train()

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        loss_ = []
        for _, (data, _) in enumerate(tqdm(train_loader, leave=False)):
            v = data.view(-1, 784 // (SCALE ** 2))
            v_gibbs = model(data.view(-1, 784 // (SCALE ** 2)))
            loss = model.free_energy(v) - model.free_energy(v_gibbs)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            model.rbm.mask_grad()
            train_op.step()

        eval(model, epoch)

        pbar.set_description_str(f'Epoch {epoch:2d} | Loss={np.mean(loss_):.4f}')

    return model

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

train_dataset = datasets.MNIST('./data',
    train=True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((28 // SCALE, 28 // SCALE)),
        lambda x: 2 * (x > 0).float() - 1
    ])
)

batch_size = 128
n_v = 784 // (SCALE ** 2)
n_h = 784 // (SCALE ** 2)
n_epochs = 20
lr = 0.01

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

rbm = TreeRBM(n_v=n_v, n_h=n_h)
model = ContrastiveDivergence(rbm, k=1)
# model = StochasticLocalization(rbm, L=10, delta=0.1)

# Train the model
model = train(model, train_loader, n_epochs=n_epochs, lr=lr)

# Save the model
torch.save(rbm.state_dict(), 'model.pt')