import numpy as np
import torch


def noise_from_spectrum(magnitude):
    '''
    Main function for generating noise samples
    Params:
        magnitude: array of magnitudes to bse used for creating noise
    '''
    magnitude = np.asarray(magnitude, dtype=np.float64)

    phases = np.exp(1j * 2 * np.pi * np.random.rand(len(magnitude)))
    spectrum = magnitude * phases

    noise = np.fft.irfft(spectrum)
    noise /= np.max(np.abs(noise) + 1e-8)

    return torch.Tensor(noise)


def generate_white_noise(length: int):
    '''
    White noise geenrating function
    
    Params:
        length (int): length of the desired noise sample
    '''
    spectrum = np.ones(length // 2 + 1)
    return noise_from_spectrum(spectrum)


def generate_pink_noise(length: int):
    '''
    Pink noise geenrating function
    
    Params:
        length (int): length of the desired noise sample
    '''
    freqs = np.fft.rfftfreq(length)
    spectrum = 1.0 / np.maximum(freqs, 1e-6)
    return noise_from_spectrum(spectrum)


def generate_band_noise(length: int, minimum: float, maximum: float, sr: int=16000):
    '''
    Noise geenrating function for a range of frequencies
    
    Params:
        length (int): length of the desired noise sample
        minimum (float): lower bound of frequencies with added noise
        maximum (float): upper bound of frequencies with added noise
    '''
    freqs = np.fft.rfftfreq(length, 1 / sr)
    spectrum = np.zeros_like(freqs)
    spectrum[(freqs >= minimum) & (freqs <= maximum)] = 1.0
    return noise_from_spectrum(spectrum)


def add_noise_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float):
    '''
    Function to add noise to a clean audio to obtain a chosen snr value
    Params:
        snr_db (float): snr value to achieved after adding noise
    Input:
        clean (Tensor): clean audio sample
        noise (Tensor): noise sample
    Output:
        noisy audio sample
    '''
    clean, noise = clean.numpy(), noise.numpy()
    clean = clean.astype(np.float64)
    noise = noise.astype(np.float64)

    min_len = min(len(clean), len(noise))
    clean = clean[:min_len]
    noise = noise[:min_len]

    rms_clean = np.sqrt(np.mean(clean**2))
    rms_noise = np.sqrt(np.mean(noise**2))

    desired_rms_noise = rms_clean / (10**(snr_db / 20))
    noise = noise * (desired_rms_noise / (rms_noise + 1e-8))

    return torch.Tensor(clean + noise)