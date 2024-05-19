import numpy as np
import ruamel.yaml
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

from signal import smooth_signal, random_noise

with open("config.yaml", "r") as stream:
    try:
        yaml = ruamel.yaml.YAML(typ='rt')
        settings = yaml.load(stream)

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
