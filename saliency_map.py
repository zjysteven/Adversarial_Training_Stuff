# https://github.com/hs2k/pytorch-smoothgrad/blob/95426b18f178558c6b6572b24fef299b4ce7d5dc/lib/gradients.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class SmoothGrad(object):
    def __init__(self, stdev_spread=0.1, n_samples=50, magnitude=True):
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitutde = magnitude

    def __call__(self, model, x, idx=None):
        '''
        model: 
            The model to be evaluated
        batch_x:
            A single image
        idx:
            Class index that are of interest.
            If nothing is provided, then the highest score class will be used.
        '''
        mode = model.training
        model.train(False)

        stdev = self.stdev_spread * (torch.max(x) - torch.min(x))
        noise_shape = list(x.shape)
        noise_shape[0] = self.n_samples
        noises = torch.normal(0, stdev, noise_shape).cuda()

        x_noisy = x + noises
        x_noisy.requires_grad = True

        logits = model(x_noisy)
        if idx is None:
            with torch.no_grad():
                _, idx = model(x).max(1)
        
        one_hot_mask = torch.zeros((1, logits.shape[-1])).cuda()
        one_hot_mask.scatter_(1, idx.unsqueeze(1), 1)
        one_hot_mask = one_hot_mask.repeat(len(x_noisy),1)
        grad = torch.autograd.grad(logits, [x_noisy], one_hot_mask)[0]
        avg_gradients = torch.mean(grad, dim=0, keepdim=True)
        model.train(mode)

        return avg_gradients


def normalize_grad(all_grad, negative=False, to_gray=False):
    n_grad = []
    for i in range(len(all_grad)):
        grad = all_grad[i:i+1]
        assert grad.shape[1] == 3
        if to_gray:
            grad = torch.sum(grad, dim=1)
        if negative:
            span = abs(grad.max().item())
            vmax = span
            vmin = -span
            th = -1
        else:
            vmax = grad.max().item()
            vmin = grad.min().item()
            th = 0
        normalized = torch.clamp((grad - vmin) / (vmax - vmin), th, 1)
        n_grad.append(normalized)
    n_grad = torch.cat(n_grad)
    return n_grad


# https://github.com/PAIR-code/saliency/blob/master/saliency/visualization.py
# If the sign of the value given by the saliency mask is not important, 
# then use VisualizeImageGrayscale, otherwise use VisualizeImageDiverging. 
# See the SmoothGrad paper for more details on which visualization method to use.

# Marginalizes across the absolute value of each channel to create a 2D single channel image, 
# and clips the image at the given percentile of the distribution. 
# This method returns a 2D tensor normalized between 0 to 1.
def VisualizeImageGrayscale(image_3d, percentile=99):
    """Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=0, and then
    clips values at a given percentile.
    """
    image_2d = np.sum(np.abs(image_3d), axis=0)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)


# Marginalizes across the value of each channel to create a 2D single channel image, 
# and clips the image at the given percentile of the distribution. 
# This method returns a 2D tensor normalized between -1 to 1 where zero remains unchanged.
def VisualizeImageDiverging(image_3d, percentile=99):
    """Returns a 3D tensor as a 2D tensor with positive and negative values.
    """
    image_2d = np.sum(image_3d, axis=0)

    span = abs(np.percentile(image_2d, percentile))
    vmin = -span
    vmax = span

    return np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)