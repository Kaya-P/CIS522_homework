from typing import List
import math
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Our implentation of a scheduler
    """

    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        batch_size=1,
        num_epochs=1,
        initial_learning_rate=1,
        initial_weight_decay=1,
    ):
        """
        Create a new scheduler.
        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.
        """
        # ... Your Code
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.initial_learning_rate = initial_learning_rate
        self.initial_weight_decay = initial_weight_decay
        self.eta_min = 0.0001

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:

        """****** Baseline ******"""
        # return [i for i in self.base_lrs]

        """config 1"""
        # return [i - (i - 1) * 0.0001 for i in self.base_lrs]

        """config 2"""
        # return [i - (i - 1) * 0.0001 for i in self.base_lrs]

        eta_min = 0.001
        T_i = len(self.base_lrs)
        num_lr = len(self.base_lrs)
        steps = int(num_lr / self.num_epochs)
        return [
            eta_min
            + (self.initial_learning_rate - eta_min)
            * (1 + math.cos(math.pi * T_cur / T_i))
            / 2
            for T_cur in range(1, self.num_epochs + 1)
            for i in range(steps)
        ]
