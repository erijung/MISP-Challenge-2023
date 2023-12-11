from transformers import AutoProcessor, ASTModel, ASTFeatureExtractor, ASTConfig
import torch
import torchaudio
from torch import nn
from utils.IRM_Calculator import ShortTimeFourierTransform

class AST_IRM_Speech_Enhancer(nn.Module):
    def __init__(self, ast_model, encoder_hidden_size = 768, output_size= 768, hop_size = 160):
        super().__init__()
        self.hop_size = hop_size
        self.encoder = ast_model
        self.processor = ASTFeatureExtractor()
        self.decoder = TransformerIRMDecoder(input_size = encoder_hidden_size, output_size=output_size)
        self.stft = ShortTimeFourierTransform()

        # add extra length
        # self.enc_config = ASTConfig(max_length=4800)
        # frequency_out_dimension, time_out_dimension = self.encoder.embeddings.get_shape(self.enc_config)
        # num_patches = frequency_out_dimension * time_out_dimension
        # extra_embeddings =  torch.zeros(1, num_patches-1212, encoder_hidden_size)
        # self.encoder.embeddings.position_embeddings = nn.Parameter(torch.concat((self.encoder.embeddings.position_embeddings, extra_embeddings), dim = 1), requires_grad=True)

    def forward(self, x, L):
        wave = self.stft(x[:, 0, :])
        x=x.cpu()
        if x.size()[1] == 1:
            x = x.squeeze(1)
        x=x.numpy()
        
        #print(x.size())
        x = self.processor(x, sampling_rate=16000, return_tensors="pt")
        # print(type(x))
        if torch.cuda.is_available():
             x = {key: value.to('cuda') for key, value in x.items()}

        out = self.encoder(**x).last_hidden_state
        if torch.cuda.is_available():
            out = out.to('cuda')
        out = self.decoder(out, L, wave)

        return out


class TransformerIRMDecoder(nn.Module):
    def __init__(self, input_size, output_size, num_decoder_layers=2, nhead=16, dim_feedforward=4096, hop_size = 160):
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
        self.hop_size = hop_size
        self.proj = nn.Linear(output_size, input_size)
    def forward(self, encoder_output, L, wave):
        # Assuming encoder_output is of shape (batch_size, seq_len, features)
        # Transformer decoder expects input of shape (seq_len, batch_size, features)
        encoder_output = encoder_output.permute(1, 0, 2)

        # Create a dummy target sequence for Transformer decoder
        target_seq_len = encoder_output.shape[0]
        batch_size = encoder_output.shape[1]
        wave = torch.abs(wave) ** 2
        if wave.shape[-1] == 2:
            wave = torch.sqrt(wave[:, :, :, 0].pow(2) + wave[:, :, :, 1].pow(2))
        wave = wave.permute(2, 0, 1)
        wave = self.proj(wave)
        # dummy_target = torch.zeros(1+L//self.hop_size, batch_size, encoder_output.shape[2])
        # if torch.cuda.is_available():
        #     dummy_target = dummy_target.to('cuda')
        # Apply the Transformer decoder
        decoder_output = self.transformer_decoder(wave, encoder_output)

        # Reshape and apply the final fully connected layer
        decoder_output = decoder_output.permute(1, 0, 2)  # Back to (batch_size, seq_len, features)
        irm_prediction = self.fc_out(decoder_output)
        irm_prediction = self.sigmoid(irm_prediction)

        return irm_prediction
