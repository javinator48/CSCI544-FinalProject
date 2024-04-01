import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
from torchcrf import CRF

from utils import (
    read_data,
    get_glove_embedding,
    LABEL
)

class NERDataset_glove(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sentence = self.X[idx]
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return (sentence, label)

class My_Glove_BLSTM_CRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(102, 256, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.33)
        self.linear = nn.Linear(512, 128)
        self.elu = nn.ELU()
        # Note: Change the number of tags (9)
        self.classifier = nn.Linear(128, 9)
        self.crf = CRF(9, batch_first=True)


    def forward(self, x, seq_len, labels=None, mask=None):
        packed_embeds = nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        x_lstm, _ = self.lstm(packed_embeds)
        x_lstm, _ = nn.utils.rnn.pad_packed_sequence(x_lstm, batch_first=True)
        x_features = self.lstm_dropout(x_lstm)
        x_features = self.linear(x_features)
        x_features = self.elu(x_features)
        emissions = self.classifier(x_features)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask)
            return loss
        else:
            return self.crf.decode(emissions)


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=LABEL.index("O"))  # Assuming "O" is a valid label and used as padding here
    mask = torch.zeros(padded_labels.size(), dtype=torch.uint8, device=padded_labels.device)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return padded_sequences, padded_labels, mask

def calculate_accuracy(model, dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    correct, count = 0, 0
    with torch.no_grad():
        for batch_x, batch_y, mask in dataloader:  # Assuming mask or seq_len is available here
            batch_x, batch_y, mask = batch_x.to(device), batch_y.to(device), mask.to(device)
            seq_len = torch.sum(mask, dim=1)  # Calculate sequence lengths from the mask
            decoded_paths = model(batch_x, seq_len)  # Use seq_len if your model needs it
            
            for i, path in enumerate(decoded_paths):
                actual_length = seq_len[i].item()  # Get the actual sequence length for this example
                true_labels = batch_y[i, :actual_length].cpu().numpy()  # Trim the true labels to the actual length
                pred_labels = path[:actual_length]  # Similarly, trim the predicted path
                
                # Now you can safely compare true_labels and pred_labels
                correct += (true_labels == pred_labels).sum()
                count += actual_length
    return correct / count if count > 0 else 0, correct, count


def train(epochs):
    with open("glove.6B.100d",  encoding='utf-8') as f:
        glove_dict = {}
        while True:
            line = f.readline()
            if not line:
                break
            line = line[:-1].split(' ')
            glove_dict[line[0]] = torch.tensor(list(map(float, line[1:])))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_sentences, train_labels = read_data("data/train")
    dev_sentences, dev_labels = read_data("data/dev")
    train_X = [get_glove_embedding(sentence, glove_dict) for sentence in train_sentences]
    dev_X = [get_glove_embedding(sentence, glove_dict) for sentence in dev_sentences]
    train_y = [[LABEL.index(l) for l in labels] for labels in train_labels]
    dev_y = [[LABEL.index(l) for l in labels] for labels in dev_labels]
    train_set = NERDataset_glove(train_X, train_y)
    dev_set = NERDataset_glove(dev_X, dev_y)
    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn
    )
    model = My_Glove_BLSTM_CRF().to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y, mask in train_loader:
            batch_x, batch_y, mask = batch_x.to(device), batch_y.to(device), mask.to(device)
            lengths = torch.sum(mask, dim=1)
            # pred = model(batch_x, lengths)
            # pred = pred.view(-1, pred.shape[-1])
            # labels = batch_y.view(-1)
            # 这里直接使用模型返回损失
            loss = model(batch_x, lengths, labels=batch_y, mask=mask)  # Pass the mask to the model
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        scheduler.step()
        print(f"Training acc after epoch {epoch}: {calculate_accuracy(model, train_loader)[0]}")
        print(f"Validation acc after epoch {epoch}: {calculate_accuracy(model, dev_loader)[0]}")
        print(f"Loss on epoch {epoch}: {epoch_loss}")
    torch.save(model, "save/blstm3.pt")


def predict(model_path):
    with open("glove.6B.100d", encoding='utf-8') as f:
        glove_dict = {}
        while True:
            line = f.readline()
            if not line:
                break
            line = line[:-1].split(' ')
            glove_dict[line[0]] = torch.tensor(list(map(float, line[1:])))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_sentences, train_labels = read_data("data/train")
    dev_sentences, dev_labels = read_data("data/dev")
    test_sentences, _ = read_data("data/test", False)
    train_X = [get_glove_embedding(sentence, glove_dict) for sentence in train_sentences]
    dev_X = [get_glove_embedding(sentence, glove_dict) for sentence in dev_sentences]
    test_X = [get_glove_embedding(sentence, glove_dict) for sentence in test_sentences]
    model = torch.load(model_path).to(device)
    model.eval()

    # Predictions for dev set
    with torch.no_grad(), open("save/dev3.out", 'w') as f_dev:
        for sentence, x in zip(dev_sentences, dev_X):
            x_tensor = torch.tensor(x, dtype=torch.float).unsqueeze(0).to(device) # Add batch dimension
            seq_len = [len(sentence)] # Actual length of sequence
            decoded_paths = model(x_tensor, seq_len)
            for path in decoded_paths:
                for j, (word, label_idx) in enumerate(zip(sentence, path)):
                    f_dev.write(f"{j+1} {word} {LABEL[label_idx]}\n")
                f_dev.write("\n") # Separate sentences by newline

    # Predictions for test set
    with torch.no_grad(), open("save/test3.out", 'w') as f_test:
        for sentence, x in zip(test_sentences, test_X):
            x_tensor = torch.tensor(x, dtype=torch.float).unsqueeze(0).to(device) # Add batch dimension
            seq_len = [len(sentence)] # Actual length of sequence
            decoded_paths = model(x_tensor, seq_len)
            for path in decoded_paths:
                for j, (word, label_idx) in enumerate(zip(sentence, path)):
                    f_test.write(f"{j+1} {word} {LABEL[label_idx]}\n")
                f_test.write("\n") # Separate sentences by newline



if __name__ == "__main__":
    train(15)
    predict("save/blstm3.pt")