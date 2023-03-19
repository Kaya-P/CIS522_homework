from typing import Callable
import torch
import torch.optim
import torch.nn as nn

# from torchvision.transforms import Compose, ToTensor,, Normalize
from torchvision.transforms import Compose, ToTensor, Normalize


class CONFIG:
    """
    config fucntion
    """

    batch_size = 128
    num_epochs = 8

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=1e-3)
    # [ToTensor(), Normalize(mean=[0.485, 0.456, 0.4], std=[0.229, 0.224, 0.2])]
    transforms = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.4], std=[0.229, 0.224, 0.2])]
    )
