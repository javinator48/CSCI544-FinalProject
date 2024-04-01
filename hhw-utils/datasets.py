import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CoNLL2003Dataset(Dataset):
    def __init__(self, data, word2idx, tag2idx) -> None:
        super().__init__()
        self.data = data
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.numerize()

    def numerize(self):
        self.num_data = []
        for words, tags in self.data:
            num_words = [self.word2idx.get(word, self.word2idx['<unk>']) for word in words]
            num_tags = [self.tag2idx.get(tag) for tag in tags]
            self.num_data.append((num_words, num_tags))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        word_seq, tag_seq = self.num_data[index]
        word_text, _ = self.data[index]
        return torch.tensor(word_seq), torch.tensor(tag_seq), word_text
    
def collate_fn(batch):
    word_seqs, tag_seqs, word_texts = zip(*batch)
    word_seqs = pad_sequence(word_seqs, batch_first=True, padding_value=0)
    tag_seqs = pad_sequence(tag_seqs, batch_first=True, padding_value=0)
    return word_seqs, tag_seqs, word_texts

def collate_fn_char(batch, char2idx):
    word_seqs, tag_seqs, word_texts = zip(*batch)
    word_seqs = pad_sequence(word_seqs, batch_first=True, padding_value=0)
    tag_seqs = pad_sequence(tag_seqs, batch_first=True, padding_value=0)

    B = word_seqs.shape[0]
    T = max([len(x) for x in word_texts])
    L = max([len(word) for words in word_texts for word in words])
    char_seqs = torch.zeros(B, T, L).int()
    for i in range(B):
        for j in range(len(word_texts[i])):
            for k in range(len(word_texts[i][j])):
                ch = word_texts[i][j][k]
                char_seqs[i][j][k] = char2idx.get(ch, char2idx['<unk>'])
    return word_seqs, tag_seqs, word_texts, char_seqs

def collate_fn_char_transformer(batch, char2idx):
    word_seqs, tag_seqs, word_texts = zip(*batch)
    word_seqs = pad_sequence(word_seqs, batch_first=True, padding_value=0)
    tag_seqs = pad_sequence(tag_seqs, batch_first=True, padding_value=0)

    B = word_seqs.shape[0]
    T = max([len(x) for x in word_texts])
    L = max([len(word) for words in word_texts for word in words])
    char_seqs = torch.zeros(B, T, L).int()
    for i in range(B):
        for j in range(len(word_texts[i])):
            for k in range(len(word_texts[i][j])):
                ch = word_texts[i][j][k]
                char_seqs[i][j][k] = char2idx.get(ch, char2idx['<unk>'])
    return word_seqs, tag_seqs, word_texts, char_seqs