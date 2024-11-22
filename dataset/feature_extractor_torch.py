import librosa
import torch
from basic_pitch.inference import predict

class FeatureExtractorTorch:
    def __init__(self, sample_rate=16000, frame_rate=250):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        return
    def compute_loudness_torch(self, audio,
                   n_fft=512,
                   range_db=80,
                   ref_db=0.0,
                   padding='constant'):
            """Perceptual loudness (weighted power) in dB using PyTorch.
            Adopted from DDSP: https://github.com/magenta/ddsp/blob/main/ddsp/spectral_ops.py
            Args:
            audio: PyTorch tensor. Shape [batch_size, audio_length] or [audio_length,].
            sample_rate: Audio sample rate in Hz.
            frame_rate: Rate of loudness frames in Hz.
            n_fft: FFT window size.
            range_db: Sets the dynamic range of loudness in decibels.
            ref_db: Sets the reference maximum perceptual loudness.
            padding: padding mode for torch.nn.functional.pad.: 'constant', 'reflect', 'replicate', 'circular'. Default is 'constant'.

            Returns:
            Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
            """
            # Ensure audio is a 2D tensor
            if len(audio.shape) == 1:
              audio = audio.unsqueeze(0)

            # Pad audio
            hop_size = self.sample_rate // self.frame_rate
            pad_amount = (n_fft - hop_size) // 2
            audio = torch.nn.functional.pad(audio, (pad_amount, pad_amount), mode=padding)

            # Compute STFT
            stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_size, win_length=n_fft, return_complex=True)
            power = stft.abs() ** 2

            # Perceptual weighting
            frequencies = torch.linspace(0, self.sample_rate // 2, n_fft // 2 + 1)
            a_weighting = torch.tensor(librosa.A_weighting(frequencies.numpy()), device=audio.device)
            a_weighting = 10 ** (a_weighting / 10)
            power = power
            a_weighting = a_weighting.unsqueeze(-1)
            power = a_weighting * power
            # Average over frequencies (weighted power per bin)
            avg_power = power.mean(dim=-1)
            loudness = 10 * torch.log10(avg_power + 1e-10)  # Convert to dB

            # Normalize loudness
            loudness = loudness - ref_db
            loudness = torch.clamp(loudness, min=-range_db)

            return loudness
    
    def get_f0(self, audio):
        return