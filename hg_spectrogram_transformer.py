from transformers import AutoProcessor, ASTModel
import torch
import torchaudio
from torch import nn, optim
from data.GssAudioDataset import GssSimulatedAudioDataset
from torch.utils.data import DataLoader
from utils.MISP_network_feature_extract import IdealMask
from models.AST_IRM import AST_IRM_Speech_Enhancer

def train(model, dataloader, criterion, optimizer, num_epochs, device='cpu'):
    wav_to_irm = IdealMask()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch in dataloader:
            inputs, irm_target = batch['waveform'], batch['label']

            # Move data to the same device as the model
            inputs = inputs.to(device)
            print('target wav batch', irm_target.shape)
            irm_target = wav_to_irm(irm_target)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            
            irm_predictions = model(inputs)
            
            print('pred shape', irm_predictions.shape)
            print(irm_target)
            print('target shape', irm_target[0].shape)
            # Compute loss
            loss = criterion(irm_predictions, irm_target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

    print('Finished Training')

def pad_waveform(waveform, target_length):
    padding_amount = target_length - waveform.size(1)
    if padding_amount > 0:
        return torch.nn.functional.pad(waveform, (0, padding_amount))
    return waveform

def collate_fn(batch):
    # Find the longest waveform in the batch
    max_length = max(map(lambda x: x['waveform'].shape[1], batch))

    # Pad each waveform in the batch to this length
    for i in range(len(batch)):
        batch[i]['waveform'] = pad_waveform(batch[i]['waveform'], max_length)
        batch[i]['label'] = pad_waveform(batch[i]['label'], max_length)

    # Stack all inputs and labels
    inputs = torch.stack([item['waveform'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {'waveform': inputs, 'label': labels}


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training on device', device)
    # Load your dataset
    dataset = GssSimulatedAudioDataset('/root/autodl-tmp/exp/gss/train/enhanced_wav/', '/root/autodl-tmp/simul-data/label/audio')  # Adjust with your dataset
    dataloader = DataLoader(dataset, batch_size=16,collate_fn=collate_fn, shuffle=True)

    # Model initialization
    processor = AutoProcessor.from_pretrained("/root/autodl-tmp/pretrained_models/pretrained/MIT/ast-finetuned-audioset-10-10-0.4593", local_files_only=True)
    ast_model = ASTModel.from_pretrained("/root/autodl-tmp/pretrained_models/pretrained/MIT/ast-finetuned-audioset-10-10-0.4593", local_files_only=True)
    model = AST_IRM_Speech_Enhancer(ast_model = ast_model, processor=processor, output_size=128).to(device)  # Adjust output_size

    # Freeze AST model encoder weights
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.decoder.parameters(), lr=0.001)  # Optimizer for decoder only

    # Training parameters
    num_epochs = 10

    # Start training
    train(model, dataloader, criterion, optimizer, num_epochs, device = device)

if __name__ == '__main__':
    main()
