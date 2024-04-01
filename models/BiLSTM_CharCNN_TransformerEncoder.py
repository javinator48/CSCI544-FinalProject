import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class BiLSTM_CharCNN_TransformerEncoder(nn.Module):
    def __init__(
            self, 
            vocab_size, char_size, embedding_dim, char_embedding_dim,
            char_final_dim, case_final_dim,
            lstm_hidden_dim, lstm_num_layers, lstm_dropout, 
            transformer_hidden_dim, transformer_num_layers, transformer_dropout,
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

        self.lstm = nn.LSTM(
            input_size=char_final_dim + embedding_dim + case_final_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(lstm_dropout)

        self.fc0 = nn.Linear(2 * lstm_hidden_dim, transformer_hidden_dim)
        self.transformer_layer = TransformerEncoderLayer(d_model=transformer_hidden_dim, nhead=4,
                                                         dim_feedforward=transformer_hidden_dim,
                                                         dropout=transformer_dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=transformer_num_layers)

        self.fc1 = nn.Linear(transformer_hidden_dim, linear_output_dim)
        self.fc2 = nn.Linear(linear_output_dim, label_size)

    def forward(self, x, case_x, char_x, seq_len):
        x = self.embedding(x)
        case_x = self.case_embedding(case_x)
        char_x = self.char_embedding(char_x)

        B, T, L, D = char_x.shape
        char_x = char_x.view(B * T, L, D).permute(0, 2, 1)
        char_x = torch.max(F.relu(self.char_cnn(char_x), inplace=True), dim=-1)[0]
        char_x = char_x.view(B, T, -1)  

        x = torch.cat((case_x, x, char_x), dim=-1)

        seq_len = torch.tensor(seq_len)
        x = pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        x = self.dropout(self.fc0(x))
        x = self.transformer_encoder(x.transpose(0, 1))
        x = x.transpose(0, 1)

        x = F.elu(self.fc1(x), alpha=0.01, inplace=True)
        x = self.fc2(x)
        return x