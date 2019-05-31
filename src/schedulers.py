from torch.optim.lr_scheduler import _LRScheduler
import numpy as np



class Cycle_LR(_LRScheduler):
    def __init__(self, optimizer, lr_factor, cycle_len, cycle_factor=2, gamma=None, jump=None, \
                 last_epoch=-1, surface=None, momentum_range=None):
        self.lr_factor = lr_factor
        self.cycle_len = cycle_len
        self.cycle_factor = cycle_factor
        self.gamma = gamma
        self.jump = jump
        self.last_epoch = last_epoch
        self.dec = True
        self.stage_epoch_count = 0
        self.current_cycle = cycle_len
        self.momentum_range = momentum_range
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.surface = surface
        if not surface:
            self.surface = self.surface_f

        for group in self.optimizer.param_groups:
            group['lr'] *= self.lr_factor

        if self.momentum_range:
            for group in self.optimizer.param_groups:
                group['momentum'] = momentum_range[1]

    @staticmethod
    def surface_f(x):
        return(np.tanh(- 2 *( 2 * x -0.8)) +1) / 2

    def switch(self):
        if self.jump:
            if self.dec:
                self.current_cycle = self.jump
                self.cycle_len *= self.cycle_factor
                if self.gamma:
                    self.lr_factor *= self.gamma
            else:
                self.current_cycle = self.cycle_len
            self.dec = not self.dec

    def get_lr(self):
        if self.stage_epoch_count > self.current_cycle:
            self.switch()
            self.stage_epoch_count = 0

        percent = self.stage_epoch_count / self.current_cycle
        if self.dec:
            factor = 1 + self.surface(percent) * (self.lr_factor - 1)
        else:
            factor = 1 + percent * (self.lr_factor - 1)

        self.stage_epoch_count += 1
        return [factor] * len(self.optimizer.param_groups)

    def get_momentum(self):
        percent = self.stage_epoch_count / self.current_cycle
        if self.dec:
            momentum = self.momentum_range[1] + percent * (
                    self.momentum_range[0] - self.momentum_range[1])
        else:
            momentum = self.momentum_range[0] + percent * (
                    self.momentum_range[1] - self.momentum_range[0])
        return [momentum] * len(self.optimizer.param_groups)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = param_group['initial_lr'] * lr

        if self.momentum_range:
            for param_group, m in zip(self.optimizer.param_groups, self.get_momentum()):
                param_group['momentum'] = m


