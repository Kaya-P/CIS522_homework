# import torch.nn as nn
# import torch.nn.functional as F
import torch


class Model(torch.nn.Module):
    """
    A simple CNN with 2 convolutional layers and 2 fully-connected layers.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        """
        init function
         """
        self.conv1 = torch.nn.Conv2d(num_channels, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, num_classes)

        # 3,chaneel, 10 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function
        """
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
