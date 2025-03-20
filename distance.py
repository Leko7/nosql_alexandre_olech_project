import torch

def lorentzian(x, y):
    return torch.sum(torch.log(1 + torch.abs(x[:, None, :] - y[None, :, :])), dim=-1)

def manhattan(x, y):
    return torch.sum(torch.abs(x[:, None, :] - y[None, :, :]), dim=-1)

def avg_l1_linf(x, y):
    abs_diff = torch.abs(x[:, None, :] - y[None, :, :])
    return (torch.sum(abs_diff, dim=-1) + torch.amax(abs_diff, dim=-1)) / 2

def jaccard(x, y):
    num = torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=-1)
    denom = torch.sum(x[:, None, :] ** 2 + y[None, :, :] ** 2 - x[:, None, :] * y[None, :, :], dim=-1) + 1e-10
    return num / denom

def minkowski(x, y, p=3):
    return torch.sum(torch.abs(x[:, None, :] - y[None, :, :]) ** p, dim=-1) ** (1 / p)