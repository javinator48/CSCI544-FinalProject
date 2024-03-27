import torch.nn as nn
import torch.nn.functional as F
import torch
class CharacterCNN(nn.Module):
    def __init__(self, input_dim, output_dim, max_char_length, num_filters=100, kernel_sizes=(3, 4, 5)):
        super(CharacterCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (kernel_size, input_dim)) for kernel_size in kernel_sizes
        ])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.max_char_length = max_char_length

    def forward(self, x):
        # x: (batch_size, sequence_length, max_char_length, input_dim)
        batch_size = x.size(0)
        x = x.view(-1, 1, self.max_char_length, x.size(3))  # reshape for Conv2d
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # conv and relu
        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]  # max pooling
        x = torch.cat(x, 1)  # concatenate all pooled features
        x = x.view(batch_size, -1, x.size(1))  # reshape to original size
        return x


class BiLSTMNER(nn.Module):
    def __init__(self, vocab_size, target_size, char_vocab_size, char_embedding_dim=30, char_hidden_dim=30,
                 embedding_dim=100, lstm_hidden_dim=256, lstm_layers=1, lstm_dropout=0.33, linear_dim=128):
        super().__init__()
        self.dropout = nn.Dropout(0.33)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.char_cnn = CharacterCNN(input_dim=char_embedding_dim, output_dim=char_hidden_dim, max_char_length=10)
        self.lstm = nn.LSTM(400, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True,
                            bidirectional=True)  # dropout=lstm_dropout)
        self.linear = nn.Linear(lstm_hidden_dim * 2, linear_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_dim, target_size)

    def forward(self, words, chars):
        word_embeds = self.dropout(self.embedding(words))
        char_embeds = self.dropout(self.char_embedding(chars))
        char_cnn_out = self.char_cnn(char_embeds)
        word_char_embeds = torch.cat((word_embeds, char_cnn_out), dim=2)

        lstm_out, _ = self.lstm(word_char_embeds)
        lstm_out = self.dropout(lstm_out)
        linear_out = self.linear(lstm_out)
        elu_out = self.elu(linear_out)
        output = self.classifier(elu_out)
        return output