import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import torch


def smooth_signal(t):
    return np.sin(t)


def random_noise(size, seed, mu, sigma):
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size)


with open("config.yaml", "r") as stream:
    try:
        settings = (yaml.safe_load(stream))

    except yaml.YAMLError as exc:
        print(exc)
        exit(1)

# Settings for random distribution
seed = settings["seed"]
mu = settings["distribution"]["mu"]
sigma = settings["distribution"]["sigma"]

# Settings for the timeseries
series = settings["timeseries"]
t = np.arange(series["start"], series["end"], series["resolution"])
size = t.size

signal = smooth_signal(t)
noise = random_noise(size, seed, mu, sigma)
noisy_signal = signal + noise

plt.plot(t, noisy_signal)

fft_signal = fft(noisy_signal)
fft_x = fftfreq(size, series["resolution"])

plt.figure()
plt.semilogy(fft_x, abs(fft_signal))
plt.show(block=True)
