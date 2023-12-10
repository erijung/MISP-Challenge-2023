from transformers import AutoProcessor, ASTModel
import torch
import torchaudio
from torch import nn
from data import GssSimulatedAudioDataset
from torch.utils.data import DataLoader
from utils import IdealMask
from models import AST_IRM_Speech_Enhancer

def train(model, dataloader, criterion, optimizer, num_epochs, freeze_encoder = True):
    wav_to_irm = IdealMask()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch in dataloader:
            inputs, irm_target = batch['waveform'], batch['label']

            # Move data to the same device as the model
            inputs = inputs.to(device)
            irm_target = wav_to_irm(irm_target)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            irm_predictions = model(inputs)
            
            # Compute loss
            loss = criterion(irm_predictions, irm_target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

    print('Finished Training')

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load your dataset
    dataset = GssSimulatedAudioDataset('/root/autodl-tmp/exp/gss/train/enhanced_wav/', '/root/autodl-tmp/simul-data/label/audio')  # Adjust with your dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model initialization
    model = AST_IRM_Speech_Enhancer(output_size=128).to(device)  # Adjust output_size

    # Freeze AST model encoder weights
    for param in model.ast_encoder.parameters():
        param.requires_grad = False

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.irm_decoder.parameters(), lr=0.001)  # Optimizer for decoder only

    # Training parameters
    num_epochs = 10

    # Start training
    train(model, dataloader, criterion, optimizer, num_epochs)

if __name__ == '__main__':
    main()
