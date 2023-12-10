import torch
from torch import nn


class IdealMask(nn.Module):
    def __init__(self, mask_type='irm', n_fft=512, hop_length=160, win_type='hamming', win_length=None, **other_params):
        super(IdealMask, self).__init__()
        self.mask_type = mask_type
        self.hop_length = hop_length
        self.stft = ShortTimeFourierTransform(
            n_fft=n_fft, hop_length=hop_length, win_type=win_type, win_length=win_length, is_complex=False)

    def forward(self, x, length=None):
        eps = 1e-8  # Define a small epsilon to avoid division by zero

        if length is not None:
            length = torch.ceil(length / self.hop_length)

        if self.mask_type in ['irm']:
            # Assuming x has shape [batch, 2, length] where
            # x[:, 0, :] is the mixture and x[:, 1, :] is the clean signal

            # Apply STFT to the mixture and clean signals
            mixture_spectrum = self.stft(x[:, 0, :])  # Mixture
            clean_spectrum = self.stft(x[:, 1, :])  # Clean

            # Compute magnitude squared (power) for both
            mixture_power = torch.abs(mixture_spectrum) ** 2
            clean_power = torch.abs(clean_spectrum) ** 2

            # Calculate the Ideal Ratio Mask
            # Adding eps for numerical stability
            irm = torch.sqrt(clean_power / (mixture_power + eps) + eps)

            return irm, length
        else:
            raise NotImplementedError('unknown mask_type')

class ShortTimeFourierTransform(nn.Module):
    def __init__(self, n_fft, hop_length, win_type='hamming', win_length=None, is_complex=False):
        super(ShortTimeFourierTransform, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.win_type = win_type
        self.is_complex = is_complex

        # Create window
        self.window = self.create_window()

    def create_window(self):
        if self.win_type == 'hamming':
            return torch.hamming_window(self.win_length)
        elif self.win_type == 'hann':
            return torch.hann_window(self.win_length)
        # Add more window types if needed
        else:
            raise ValueError(f"Unknown window type: {self.win_type}")

    def forward(self, x):
        # Apply STFT
        stft_output = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=self.window.to(x.device), 
            return_complex=self.is_complex
        )
        return stft_output