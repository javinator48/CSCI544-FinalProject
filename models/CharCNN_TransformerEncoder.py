import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Compute the positional encodings once in log space
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        self.register_buffer('positional_embedding', self.pe)
    
    def forward(self, x):
        device = x.device
        # Add positional encoding to input
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        encoding = self.pe[:, :seq_len].to(device)
        x = x + encoding
        return x

class CharCNN_TransformerEncoder(nn.Module):
    def __init__(
            self,
            vocab_size, char_size, embedding_dim,
            char_embedding_dim, char_final_dim, case_final_dim,
            num_head, num_layer, dropout,
            linear_output_dim, label_size, embedding_mat=None
        ):  
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_mat is not None:
            self.embedding.weight.data.copy_(embedding_mat)
        
        self.case_embedding = nn.Embedding(2, case_final_dim)
        self.char_embedding = nn.Embedding(char_size, char_embedding_dim)

        self.char_cnn = nn.Sequential(
            nn.Conv1d(char_embedding_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, char_final_dim, kernel_size=3, padding=1),
        )

        self.d_model = embedding_dim + case_final_dim + char_final_dim
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=1000)

        self.encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(self.d_model, num_head),
            num_layers=num_layer,
        )

        self.fc1 = nn.Linear(self.d_model, linear_output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(linear_output_dim, label_size)

    def forward(self, x, case_x, char_x):
        x = self.embedding(x)
        case_x = self.case_embedding(case_x)
        char_x = self.char_embedding(char_x)

        B, T, L, D = char_x.shape
        char_x = char_x.view(B * T, L, D).permute(0, 2, 1)
        char_x, _ = torch.max(F.relu(self.char_cnn(char_x), inplace=True), dim=-1)
        char_x = char_x.view(B, T, -1)  # BxTxD_char
        
        x = torch.cat((case_x, x, char_x), dim=-1)
        x = self.positional_encoding(x)
        
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1)
        
        x = self.dropout(F.relu(self.fc1(x), inplace=True))
        x = self.fc2(x)
        return x