import importlib
import os
import torch
import numpy as np


def convert_to_tensor(input_obj, device):
    if isinstance(input_obj, dict):
        for key in input_obj.keys():
            input_obj[key] = convert_to_tensor(input_obj[key], device)
    elif isinstance(input_obj, np.ndarray):
        input_obj = torch.from_numpy(input_obj).type(torch.float64).to(device)
    elif isinstance(input_obj, int):
        input_obj = torch.tensor(input_obj, dtype=torch.float64).to(device)
    elif input_obj is None:
        return None
    elif isinstance(input_obj,torch.Tensor):
        input_obj = input_obj.to(device)
    else:
        raise NotImplementedError
    return input_obj

def convert_tensor_to_numpy(input_obj):
    if isinstance(input_obj, dict):
        for key in input_obj.keys():
            input_obj[key] = convert_tensor_to_numpy(input_obj[key])
    elif isinstance(input_obj, torch.Tensor):
        input_obj = input_obj.cpu().detach().numpy()
    # I'm assuming set only contains strings here.
    elif isinstance(input_obj, int) or isinstance(input_obj, float) or isinstance(input_obj, set):
        input_obj = input_obj
    elif isinstance(input_obj, np.ndarray):
        input_obj = input_obj
    elif input_obj is None:
        return None
    else:
        raise NotImplementedError
    return input_obj

