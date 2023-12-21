from torch.utils.data import Dataset
from torchaudio import load
import torch
import os
import glob
import random

class GssSimulatedAudioDataset(Dataset):
    """
    Dataset for (X: Gss Audio Files, Y: Clean Audio Files)
    """

    def __init__(self, gss_wav_dir, label_dir, file_list, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing audio files.
            label_dir (str): Path to the directory containing label files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = gss_wav_dir
        self.label_dir = label_dir
        self.transform = transform

        file_list = open(file_list, "r")
        gss_audio_files = file_list.read().split('\n')
        self.gss_audio_files = gss_audio_files
        file_list.close()
        # List all audio files in the input directory
        # Note that his is nested so {session}_{noise_lvl}db-{source}-{timestamp1}_{timestamp2}.wav
        # has full path as input_dir/{session}_{noise_lvl}db/{session}_{noise_lvl}db-{source}-{timestamp1}_{timestamp2}.wav
        # self.gss_audio_files = [os.path.join(root.split('/')[-1], f) for root, sub, flist in os.walk(self.data_dir) for f in flist]
        # print(self.gss_audio_files)
        # self.gss_audio_files = self.gss_audio_files[:1000]

    def __len__(self):
        return len(self.gss_audio_files)
    
    def truncate_data(self, samples):
        random.shuffle(self.gss_audio_files)
        self.gss_audio_files = self.gss_audio_files[:samples]

    def __getitem__(self, idx):
        # Get base file name
        input_file = self.gss_audio_files[idx]
        

        # Extract information from the input file name
        session_noise_lvl, source_timestamps = input_file.split('/')[-1].split('db-')
        session, noise_lvl = session_noise_lvl.split('_')
        source, timestamps_wav = source_timestamps.split('-') # timestamps_wav = {timestamp1}_{timestamp2}.wav

        # Construct the input and label file path
        audio_file_path = os.path.join(self.data_dir, session_noise_lvl + 'db', input_file)
        label_file_path = os.path.join(self.label_dir, f"{source}-{session}-{timestamps_wav}")
        
        # Load the audio file using torchaudio
        input_waveform, sample_rate = load(audio_file_path)
        files = None
        try:
            clean_waveform, clean_sample_rate = load(label_file_path)
        except:
            timestamp1 = timestamps_wav.split('_')[0]
            # Example usage
            directory = self.label_dir  # Replace with the path to your directory
            pattern = f'{source}-{session}-{timestamp1}'  # Replace with your actual pattern
            files = find_files(directory, pattern)

            # for file in files:
            #     print(file)
            label_file_path = os.path.join(self.label_dir, files[0].split('/')[-1])
            clean_waveform, clean_sample_rate = load(label_file_path)
        assert(sample_rate == clean_sample_rate)
        if clean_waveform.shape != input_waveform.shape:
            max_length = max([input_waveform.shape[1 ], clean_waveform.shape[1]])
            # print('padding input with', max_length - input_waveform.shape[1])
            input_waveform = pad_waveform(input_waveform, max_length)
            # print('padding target with', max_length - clean_waveform.shape[1])
            clean_waveform = pad_waveform(clean_waveform, max_length)
            input_waveform = input_waveform + clean_waveform

        # Apply any specified transforms
        if self.transform:
            waveform = self.transform(waveform)

        # Return a dictionary containing the waveform and label
        sample = {'in_file': self.gss_audio_files[idx], 'waveform': input_waveform, 'label': clean_waveform}

        return sample


class GssSimulatedAudioSpectrogramDataset(GssSimulatedAudioDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec_transform = Spectrogram(n_fft=400)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        noisy_waveform = sample['noisy_waveform']
        clean_waveform = sample['clean_waveform']

        noisy_spec = self.spec_transform(noisy_waveform)
        clean_spec = self.spec_transform(clean_waveform)

        irm = self.calculate_irm(noisy_spec, clean_spec)

        return {'noisy_waveform': noisy_waveform, 'irm': irm}

    def calculate_irm(self, noisy_spec, clean_spec):
        noise_spec = noisy_spec - clean_spec
        irm = clean_spec / (clean_spec + noise_spec + 1e-8)  # Adding a small constant for numerical stability
        return irm



def find_files(directory, pattern):
    # Construct the full pattern
    full_pattern = f"{directory}/{pattern}*"
    # Use glob to find files matching the pattern
    return glob.glob(full_pattern)


def pad_waveform(waveform, target_length):
    padding_amount = target_length - waveform.size(1)
    if padding_amount > 0:
        return torch.nn.functional.pad(waveform, (0, padding_amount))
    return waveform
def truncate_waveform(waveform, target_length):
    return waveform[:, :target_length], waveform[:, target_length:]