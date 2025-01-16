from torchsig.datasets.synthetic import CarrierWaveSpikeDataset
# placeholder for plot functions
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def plot_psd(f,pxx,title):
    plt.figure(figsize=(8, 6))
    plt.plot(f, pxx)
    plt.title(title)
    plt.xlabel('freq hz')
    plt.ylabel('power')
    plt.grid(True)
    plt.show()

def get_psd(signal, sampling_rate):
    """ This function gets the power spectral density from the signal using Welch's function. """
    freq, Pxx = sig.welch(signal, fs = sampling_rate)
    sorted_indices = np.argsort(freq)
    freq = freq[sorted_indices]
    Pxx = Pxx[sorted_indices]
    Pxx = np.log(Pxx)
    return freq, Pxx    

cw_dataset = CarrierWaveSpikeDataset(random_data=True)
signal_idx = 3
iq_samples = cw_dataset[signal_idx][0]

dbg = 1


# Set power
iq_samples = iq_samples / np.sqrt(np.mean(np.abs(iq_samples) ** 2))
iq_samples = np.sqrt(cw_dataset[signal_idx][1][0]["bandwidth"]) * (10 ** (cw_dataset[signal_idx][1][0]["snr"] / 20.0)) * iq_samples / np.sqrt(2)
    



# f, Pxx = get_psd(signal=cw_dataset[signal_idx][0], sampling_rate=1000)
f, Pxx = get_psd(signal=iq_samples, sampling_rate=1000)
plot_psd(f=f, pxx=Pxx, title="TITLE")