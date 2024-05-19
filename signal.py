import numpy as np


def smooth_signal(t):
    return np.sin(t)


def random_noise(size, seed, mu, sigma):
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size)
