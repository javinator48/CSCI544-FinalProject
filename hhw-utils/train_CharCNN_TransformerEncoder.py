import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from utils import load_data, generate_mappings_char, load_glove, generate_embedding_matrix
from datasets import CoNLL2003Dataset, collate_fn_char_transformer
from models.CharCNN_TransformerEncoder import CharCNN_TransformerEncoder

raw_train, train_words, train_tags = load_data('data/train')
raw_dev, dev_words, dev_tags = load_data('data/dev')
word2idx, tag2idx, char2idx = generate_mappings_char(raw_train, train_words, train_tags, threshold=0)
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

train_dataset = CoNLL2003Dataset(raw_train, word2idx, tag2idx)
val_dataset = CoNLL2003Dataset(raw_dev, word2idx, tag2idx)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: collate_fn_char_transformer(x, char2idx)
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda x: collate_fn_char_transformer(x, char2idx)
)

# Model settings:
vocab_size = len(word2idx)
label_size = len(tag2idx)
char_size = len(char2idx)
embedding_dim = 100
char_embedding_dim = 30
dropout = 0.33
linear_output_dim = 128
char_final_dim = 25
case_final_dim = 3
transformer_num_head = 4
transformer_hidden_dim = 256
transformer_num_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# glove_embeddings = load_glove('glove.6B.100d.txt')
# embedding_mat = generate_embedding_matrix(word2idx, embedding_dim, glove_embeddings)

model = CharCNN_TransformerEncoder(
    vocab_size, char_size, embedding_dim, char_embedding_dim, 
    char_final_dim, case_final_dim,
    transformer_num_head, transformer_num_layers, dropout,
    linear_output_dim, label_size
)
# optimizer = AdamW(model.parameters(), lr=1e-3)
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
weight = torch.ones(label_size)
weight[0] = 0   # ignores the loss on the padding token
weight[tag2idx['O']] = 0.2
weight = weight.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0, weight=weight)
epochs = 30

model = model.to(device)

best_f1 = 0
best_loss = torch.inf

for epoch in range(epochs):
    train_loss = 0
    correct = 0
    amount = 0
    model.train()
    train_loop = tqdm(train_loader, desc=f'Epoch: {epoch + 1}/{epochs}')
    cur_total = 0
    for data, target, words, char_data in train_loop:
        data = data.to(device)
        target = target.to(device)
        case_data = torch.zeros(data.shape[0], data.shape[1]).int()
        for i in range(data.shape[0]):
            for j in range(len(words[i])):
                if words[i][j] != words[i][j].lower():
                    case_data[i][j] = 1
        case_data = case_data.to(device)
        char_data = char_data.to(device)
                
        optimizer.zero_grad()
        pred = model(data, case_data, char_data)
        pred = pred.view(-1, label_size)
        target = target.view(-1)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().item() * data.shape[0]
        target = target.cpu()
        pred = torch.argmax(pred, dim=1).cpu()
        mask = target != 0
        masked_pred = pred[mask]
        masked_target = target[mask]
        correct += sum(masked_pred == masked_target)
        amount += sum(mask)

        cur_total += data.shape[0]
        running_loss = train_loss / cur_total
        running_acc = correct / amount
        train_loop.set_postfix(loss=running_loss, acc=running_acc.item())

    train_loss /= len(train_dataset)
    train_acc = correct / amount

    val_loss = 0
    y_true, y_pred = [], []
    correct = 0
    amount = 0
    model.eval()
    with torch.no_grad():
        for data, target, words, char_data in tqdm(val_loader):
            data = data.to(device)
            target = target.to(device)
            case_data = torch.zeros(data.shape[0], data.shape[1]).int()
            for i in range(data.shape[0]):
                for j in range(len(words[i])):
                    if words[i][j] != words[i][j].lower():
                        case_data[i][j] = 1
            case_data = case_data.to(device)
            char_data = char_data.to(device)
                
            pred = model(data, case_data, char_data)
            pred = pred.view(-1, label_size)
            target = target.view(-1)
            loss = criterion(pred, target)

            val_loss += loss.cpu().item() * data.shape[0]

            target = target.cpu()
            pred = torch.argmax(pred, dim=1).cpu()
            mask = target != 0
            masked_pred = pred[mask]
            masked_target = target[mask]
            correct += sum(masked_pred == masked_target)
            amount += sum(mask)
            y_pred.extend(masked_pred)
            y_true.extend(masked_target)

    val_loss /= len(val_dataset)
    val_acc = correct / amount
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    print('train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(train_loss, train_acc, val_loss, val_acc))
    print('val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f}'.format(val_precision, val_recall, val_f1))

    if best_f1 < val_f1:
        checkpoint = {
            'state_dict': model.state_dict(),
            'idx2tag': idx2tag,
            'tag2idx': tag2idx,
            'word2idx': word2idx,
            'char2idx': char2idx
        }
        torch.save(checkpoint, 'blstm7.pt')
        print('update best f1 from {:.4f} to {:.4f}'.format(best_f1, val_f1))
        best_f1 = val_f1

    if best_loss > val_loss:
        checkpoint = {
            'state_dict': model.state_dict(),
            'idx2tag': idx2tag,
            'tag2idx': tag2idx,
            'word2idx': word2idx,
            'char2idx': char2idx
        }
        torch.save(checkpoint, 'blstm7_loss.pt')
        print('update best loss from {:.4f} to {:.4f}'.format(best_loss, val_loss))
        best_loss = val_loss

predict_model = CharCNN_TransformerEncoder(
    vocab_size, char_size, embedding_dim, char_embedding_dim, 
    char_final_dim, case_final_dim,
    transformer_num_head, transformer_num_layers, dropout,
    linear_output_dim, label_size
)
checkpoint = torch.load('blstm7.pt')
word2idx = dict(checkpoint['word2idx'])
tag2idx = dict(checkpoint['tag2idx'])
idx2tag = dict(checkpoint['idx2tag'])
char2idx = dict(checkpoint['char2idx'])
state_dict = checkpoint['state_dict']
predict_model.load_state_dict(state_dict)
predict_model = predict_model.to(device)

# predict dev split
with open('dev7.out', 'w') as f:
    with torch.no_grad():
        predict_model.eval()
        for data, targets, words, char_data in tqdm(val_loader):
            data = data.to(device)
            case_data = torch.zeros(data.shape[0], data.shape[1]).int()
            for i in range(data.shape[0]):
                for j in range(len(words[i])):
                    if words[i][j] != words[i][j].lower():
                        case_data[i][j] = 1
            case_data = case_data.to(device)
            char_data = char_data.to(device)
            seq_lengths = [len(x) for x in words]
                
            preds = predict_model(data, case_data, char_data)
            preds = torch.argmax(preds, dim=2).cpu()
            
            for pred, word, length in zip(preds, words, seq_lengths):
                pred = pred[:length]
                for idx in range(length):
                    f.write(f'{idx + 1} {word[idx]} {idx2tag[pred[idx].item()]}\n')
                f.write('\n')

raw_test, _, _ = load_data('data/test')
test_dataset = CoNLL2003Dataset(raw_test, word2idx, tag2idx)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda x: collate_fn_char_transformer(x, char2idx)
)

# predict test split
with open('test7.out', 'w') as f:
    with torch.no_grad():
        predict_model.eval()
        for data, _, words, char_data in tqdm(test_loader):
            data = data.to(device)
            case_data = torch.zeros(data.shape[0], data.shape[1]).int()
            for i in range(data.shape[0]):
                for j in range(len(words[i])):
                    if words[i][j] != words[i][j].lower():
                        case_data[i][j] = 1
            case_data = case_data.to(device)
            char_data = char_data.to(device)
            seq_lengths = [len(x) for x in words]
                
            preds = predict_model(data, case_data, char_data)
            preds = torch.argmax(preds, dim=2).cpu()
            
            for pred, word, length in zip(preds, words, seq_lengths):
                pred = pred[:length]
                for idx in range(length):
                    f.write(f'{idx + 1} {word[idx]} {idx2tag[pred[idx].item()]}\n')
                f.write('\n')