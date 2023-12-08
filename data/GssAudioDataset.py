from torch.utils.data import Dataset
from torchaudio import load
import os

class GssAudioDataset(Dataset):
    """
    Dataset for (X: Gss Audio Files, Y: Clean Audio Files)
    """

    def __init__(self, gss_wav_dir, label_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing audio files.
            label_dir (str): Path to the directory containing label files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = gss_wav_dir
        self.label_dir = label_dir
        self.transform = transform

        # List all audio files in the input directory
        # Note that his is nested so {session}_{noise_lvl}db-{source}-{timestamp1}_{timestamp2}.wav
        # has full path as input_dir/{session}_{noise_lvl}db/{session}_{noise_lvl}db-{source}-{timestamp1}_{timestamp2}.wav
        self.gss_audio_files = [os.path.join(root.split('/')[-1], f) for root, sub, flist in os.walk(self.data_dir) for f in flist]
        # print(self.gss_audio_files)

    def __len__(self):
        return len(self.gss_audio_files)

    def __getitem__(self, idx):
        # Get base file name
        input_file = self.gss_audio_files[idx]
        

        # Extract information from the input file name
        session_noise_lvl, source_timestamps = input_file.split('/')[-1].split('db-')
        session, noise_lvl = session_noise_lvl.split('_')
        source, timestamps_wav = source_timestamps.split('-') # timestamps_wav = {timestamp1}_{timestamp2}.wav

        # Construct the input and label file path
        audio_file_path = os.path.join(self.data_dir, input_file)
        label_file_path = os.path.join(self.label_dir, f"{source}_{session}_{timestamps_wav}")
        
        # Load the audio file using torchaudio
        input_waveform, sample_rate = load(audio_file_path)
        clean_waveform, clean_sample_rate = load(label_file_path)
        assert(sample_rate == clean_sample_rate)

        # Apply any specified transforms
        if self.transform:
            waveform = self.transform(waveform)

        # Return a dictionary containing the waveform and label
        sample = {'waveform': waveform, 'label': clean_waveform}

        return sample

