import torch
from torch import nn


class MSE():
    def __init__(self, exclude_mask, batch_mean=False):
        self._mask = ~exclude_mask
        self._batch_mean = batch_mean

    def __call__(self, pred, true, *args):
        pred = pred[:, self._mask]
        true = true[:, self._mask]
        
        error = torch.square(pred - true).mean(1)
        if self._batch_mean:
            error = error.mean()
        return error
    
    
class MAE():
    def __init__(self, exclude_mask, batch_mean=False):
        self._mask = ~exclude_mask
        self._batch_mean = batch_mean

    def __call__(self, pred, true, *args):
        pred = pred[:, self._mask]
        true = true[:, self._mask]
        
        error = torch.abs(pred - true).mean(1)
        if self._batch_mean:
            error = error.mean()
        return error


class ResidualLoss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self._loss_fn = loss_fn
    
    def forward(self, pred, true, inp):
        return self._loss_fn(pred, (true-inp))