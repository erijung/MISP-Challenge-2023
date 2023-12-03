from transformers import AutoProcessor, ASTModel
import torch
import torchaudio
# from datasets import load_datset

processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

#test how this works

waveform, sampling_rate = torchaudio.load('../../data/dev_far_audio/R80_S455456457_C01_I0_Far_0.wav') # a sample wav file (only has one channel)
print(waveform.shape)

spec_transform = torchaudio.transforms.Spectrogram(n_fft=400)
spectrogram = spec_transform(waveform)
print(spectrogram.shape)

waveform = waveform.numpy()

inputs = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(list(last_hidden_states.shape))
print(last_hidden_states)