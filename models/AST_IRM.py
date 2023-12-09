from transformers import AutoProcessor, ASTModel
import torch
import torchaudio
from torch import nn

class AST_IRM_Speech_Enhancer(nn.Module):
    def __init__(self, ast_model = "MIT/ast-finetuned-audioset-10-10-0.4593", encoder_hidden_size = 768, output_size= 768):
        self.encoder = ASTModel.from_pretrained(ast_model)
        self.processor = AutoProcessor.from_pretrained(ast_model)
        self.decoder = TransformerIRMDecoder(input_size = encoder_hidden_size, output_size=output_size)
    def forward(x):
        x = self.processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")
        out = self.encoder(x).last_hidden_state
        out = self.decoder(out)




class TransformerIRMDecoder(nn.Module):
    def __init__(self, input_size, output_size, num_decoder_layers=3, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_size, 
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, 
            num_layers=num_decoder_layers
        )
        self.fc_out = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_output):
        # Assuming encoder_output is of shape (batch_size, seq_len, features)
        # Transformer decoder expects input of shape (seq_len, batch_size, features)
        encoder_output = encoder_output.permute(1, 0, 2)

        # Create a dummy target sequence for Transformer decoder
        target_seq_len = encoder_output.shape[0]
        batch_size = encoder_output.shape[1]
        dummy_target = torch.zeros(target_seq_len, batch_size, encoder_output.shape[2])

        # Apply the Transformer decoder
        decoder_output = self.transformer_decoder(dummy_target, encoder_output)

        # Reshape and apply the final fully connected layer
        decoder_output = decoder_output.permute(1, 0, 2)  # Back to (batch_size, seq_len, features)
        irm_prediction = self.fc_out(decoder_output)
        irm_prediction = self.sigmoid(irm_prediction)

        return irm_prediction
