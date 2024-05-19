import numpy as np
import ruamel.yaml
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

from signal import smooth_signal, random_noise

with open("config.yaml", "r") as stream:
    try:
        yaml = ruamel.yaml.YAML(typ='rt')
        settings = yaml.load(stream)

    except ruamel.yaml.YAMLError as exc:
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

fft_threshold = settings["FFT"]["threshold"]

# Initializing signal and noise
signal = smooth_signal(t)
noise = random_noise(size, seed, mu, sigma)
noisy_signal = signal + noise

fft_pure = fft(signal)
fft_signal = fft(noisy_signal)
fft_x = fftfreq(size, series["resolution"])

fft_cleaned = np.where(abs(fft_signal) < fft_threshold, 0, fft_signal)
cleaned_signal = ifft(fft_cleaned)


# Plotting signal and noise
plt.plot(t, noisy_signal, label='Noisy')
plt.plot(t, cleaned_signal, label='Cleaned')
plt.plot(t, signal, '--', label='Original')
plt.legend()
plt.title("Signal with noise")
plt.xlabel("Time")
plt.ylabel("Signal")

plt.figure()
plt.semilogy(fft_x, abs(fft_signal), label="Noisy")
plt.semilogy(fft_x, abs(fft_cleaned), label='Cleaned')
plt.semilogy(fft_x, abs(fft_pure), '--', label='Pure')
plt.legend()
plt.title("FFT of signal with and without noise")
plt.xlabel("Frequency")
plt.ylabel("Occurence")


plt.figure()
plt.hist(noise)
plt.title("Distribution of noise")
plt.xlabel("Noise strength")
plt.ylabel("Occurence")

plt.show(block=True)
