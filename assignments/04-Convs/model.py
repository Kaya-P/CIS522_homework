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
        self.conv1 = torch.nn.Conv2d(num_channels, 28, 5)
        # self.pool = torch.nn.MaxPool2d(2, 2)
        # self.conv2 = torch.nn.Conv2d(4, 16, 5)
        # e
        self.fc1 = torch.nn.Linear(28 * 28 * 28, num_classes)
        # self.fc3 = torch.nn.Linear(120, num_classes)
        # slef d
        # 3,chaneel, 10 classes

    # 16, 28,28 and 16,5
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward function
        """
        x = torch.nn.functional.relu(self.conv1(x))
        # print(x.shape)
        # x = torch.nn.functional.relu(self.conv2(x))
        # print(x.shape)
        x = x.view(-1, 28 * 28 * 28)
        x = self.fc1(x)
        return x
