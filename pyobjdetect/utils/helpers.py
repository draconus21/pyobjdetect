import numpy as np
import torch


def torch2numpy(tensor: torch.Tensor):
    arr = tensor.numpy()
    if len(arr.shape) == 4:
        ch_ax = 1
    elif len(arr.shape) == 3:
        ch_ax = 0
    elif len(arr.shape) == 2:
        return arr
    else:
        raise ValueError(f"tensor has invalid shape ({arr.shape}). Must be have 2, 3 or 4 dimensions.")

    arr = np.moveaxis(arr, [ch_ax], [len(arr.shape) - 1])
    return arr
