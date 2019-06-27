import numpy as np


class CyclicLR:
    def __init__(self, warm_epoch, all_epoch, target_lr, iters_epoch, period=4, base_lr=1e-10):
        self.iter = -1
        self.warmIters = iters_epoch * warm_epoch
        self.target_lr = target_lr
        self.all_iters = iters_epoch * all_epoch
        self.step_iters = (all_epoch - warm_epoch) // period * iters_epoch
        self.period, self.base_lr = period, base_lr

    def _warm(self):
        return (self.target_lr - self.base_lr) * self.iter / self.warmIters + self.base_lr

    def _cosine(self, cur_iter, target_lr):
        lr = (target_lr - self.base_lr) * (1.0 + np.cos(np.pi * cur_iter / self.step_iters)) * 0.5 + self.base_lr
        return lr

    @property
    def lr(self):
        self.iter += 1
        if self.iter < self.warmIters:
            lr = self._warm()
        elif self.iter < self.all_iters:
            cos_iter = self.iter - self.warmIters
            cur_iter = cos_iter % self.step_iters
            target_lr = (1.0 - cos_iter // self.step_iters / self.period) * self.target_lr
            lr = self._cosine(cur_iter, target_lr)
        else:
            lr = self.base_lr
        return lr

    @staticmethod
    def update_lr(lr, opti):
        for param_group in opti.param_groups:
            param_group['lr'] = lr
