import torch
from torch import nn
from collections import OrderedDict
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class ImageFitting(Dataset):
    def __init__(self, imagesData, mean, std, sidelength):
        transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(mean, std)
        ])
        self.image = transform(Image.fromarray(imagesData)).permute(1, 2, 0).view(-1, imagesData.shape[-1])
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx >= len(self): raise IndexError
        return self.coords, self.image


class ImageFittingIdx(Dataset):
    def __init__(self, imagesData, mean, std, sidelength):
        transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(mean, std)
        ])
        self.images = []
        for i in range(len(imagesData)):
            self.images.append(transform(Image.fromarray(imagesData[i])).permute(1, 2, 0).view(-1, imagesData.shape[-1]))
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx >= len(self): raise IndexError
        return torch.cat([self.coords, torch.full((len(self.coords), 1), idx)], dim=1), self.images[idx]


def train(img_siren, dataloader, steps, device):
    steps_til_summary = int(steps / 5)
    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())
    train_losses = []
    for step in range(steps):
        batch_losses = []
        for batch_idx, batch in enumerate(dataloader):
            model_input, ground_truth = batch[0].to(device), batch[1].to(device)
            model_output, coords = img_siren(model_input)
            loss = ((model_output - ground_truth) ** 2).mean()

            if not step % steps_til_summary:
                print("Step %d, Total loss %0.6f" % (step, loss))
            optim.zero_grad()
            loss.backward()
            batch_losses.append(loss.item())
            optim.step()
        train_losses.append(np.mean(batch_losses))
    return train_losses
