import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path


class Denoiser:
    '''
    Denoiser class which consists of different denoise methods
    Params:
        alpha (float): a number for bias correction during noise estimation (normally [1.2, 2.0])
    '''

    def __init__(self, alpha: float):
        assert alpha >= 1.0, 'alpha must be more than 1.0'
        self.alpha = alpha

    def simple_denoiser(self, path: Path):
        '''
        Simple denoiser with one mask for the whole audio
        '''
        audio, sr = torchaudio.load(path) # [num_channels, T]
        audio = audio.mean(dim=0) # [T]

        spectrogram = torch.stft(audio, n_fft=400, return_complex=True) # [F, T']
        P = spectrogram.abs()**2
        noise = torch.min(P, dim=1, keepdim=True).values # [F, 1]
        noise_corrected = noise * self.alpha
        mask = (P - noise_corrected) / (P + 1e-8) 
        mask = mask.clamp(0, 1)  

        denoised_spec = spectrogram * mask
        denoised_audio = torch.istft(denoised_spec, n_fft=400)

        if denoised_audio.dim() == 1: 
            denoised_audio = denoised_audio.unsqueeze(0)

        return denoised_audio, sr
    
    def window_denoiser(self, path: Path):
        '''
        Denoiser based on windowed masking with hop_length based on sample rate of the audio
        '''
        audio, sr = torchaudio.load(path) # [num_channels, T]
        audio = audio.mean(dim=0) # [T]
        
        win_ms = 25.0
        hop_ms = 6.25
        n_fft = int(sr * win_ms / 1000)
        n_fft = max(256, n_fft)
        n_fft += n_fft % 2
        hop_length = int(sr * hop_ms / 1000)
        window_len = max(1, int(0.5 * sr / hop_length))
        window = torch.hann_window(n_fft, device=audio.device)

        spectrogram = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, 
                                    win_length=n_fft, window=window, return_complex=True)   # [F, T']
        P = spectrogram.abs()**2
        P_pad = F.pad(P, (window_len - 1, 0), mode="replicate") # padding for saving shape

        windows = P_pad.unfold(dimension=1, size=window_len, step=1) # [F, T, L]
        noises = torch.min(windows, dim=-1).values
        noises_corrected = self.alpha * noises

        mask = (P - noises_corrected) / (P + 1e-8)
        mask = mask.clamp(0, 1)
        mask = F.avg_pool2d(
            mask.unsqueeze(0).unsqueeze(0),
            kernel_size=(3, 5),   # freq and time smoothing of the mask
            stride=1,
            padding=(1, 2)
        ).squeeze(0).squeeze(0)
        mask = mask.clamp(0, 1)

        denoised_spec = spectrogram * mask
        denoised_audio = torch.istft(denoised_spec, n_fft=n_fft, hop_length=hop_length, 
                                    win_length=n_fft, window=window, length=audio.shape[-1])
        
        if denoised_audio.dim() == 1: 
            denoised_audio = denoised_audio.unsqueeze(0)

        return denoised_audio, sr