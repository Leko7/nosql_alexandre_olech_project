import numpy as np

def z_score_norm(x):
    return (x - np.mean(x)) / np.std(x)

def min_max_norm(x):
    min_value = np.min(x)
    max_value = np.max(x)

    return (x - min_value) / (max_value - min_value)

def unit_length_norm(x):
    return x / np.linalg.norm(x)

def mean_norm(x):
    min_value = np.min(x)
    max_value = np.max(x)
    mean_value = np.mean(x)

    return (x - mean_value) / (max_value - min_value)

def tanh_norm(x):
    return np.tanh(x)

def adaptive_scaling(x, y):
    alpha = np.dot(x, y.T) / np.dot(y, y.T)
    y_new = alpha * y
    x_new = x
    return x_new, y_new

def median_norm(x):
    return x / np.median(x)

def sigmoid_norm(x):
    return 1 / (1 - np.exp(-x))