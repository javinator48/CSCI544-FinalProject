import torch

def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    raw_data = []
    words = set()
    tags = set()
    cur_words, cur_tags = [], []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0:
            raw_data.append((cur_words, cur_tags))
            words.update(cur_words)
            tags.update(cur_tags)
            cur_words, cur_tags = [], []
            continue
        if len(parts) == 3:
            _, word, tag = parts
            cur_words.append(word)
            cur_tags.append(tag)
        else:
            _, word = parts
            cur_words.append(word)
    return raw_data, words, tags
    
# generate_mappings should only use the training data, because there may exists unknown words
# threshold can be used to determine which words are needed
def generate_mappings(raw_train, train_words, train_tags, threshold=0):
    filtered_words = train_words
    if threshold != 0:
        from collections import Counter
        word_cnt = Counter(word for words, _ in raw_train for word in words)
        filtered_words = [word for word, cnt in word_cnt.items() if cnt > threshold]
    
    word2idx = {word: idx + 2 for idx, word in enumerate(filtered_words)}
    word2idx['<pad>'] = 0   # use <pad> for padding word
    word2idx['<unk>'] = 1   # use <unk> for unknown words and low frequency words

    tag2idx = {tag: idx + 1 for idx, tag in enumerate(train_tags)}
    tag2idx['<pad>'] = 0    # use <pad> for padding tag

    return word2idx, tag2idx

def generate_mappings_lower(raw_train, train_words, train_tags, threshold=0):
    filtered_words = train_words
    if threshold != 0:
        from collections import Counter
        word_cnt = Counter(word.lower() for words, _ in raw_train for word in words)
        filtered_words = [word for word, cnt in word_cnt.items() if cnt > threshold]
    
    word2idx = {word: idx + 2 for idx, word in enumerate(filtered_words)}
    word2idx['<pad>'] = 0   # use <pad> for padding word
    word2idx['<unk>'] = 1   # use <unk> for unknown words and low frequency words

    tag2idx = {tag: idx + 1 for idx, tag in enumerate(train_tags)}
    tag2idx['<pad>'] = 0    # use <pad> for padding tag

    return word2idx, tag2idx

def generate_mappings_char(raw_train, train_words, train_tags, threshold=0):
    filtered_words = train_words
    if threshold != 0:
        from collections import Counter
        word_cnt = Counter(word.lower() for words, _ in raw_train for word in words)
        filtered_words = [word for word, cnt in word_cnt.items() if cnt > threshold]
    
    word2idx = {word: idx + 2 for idx, word in enumerate(filtered_words)}
    word2idx['<pad>'] = 0   # use <pad> for padding word
    word2idx['<unk>'] = 1   # use <unk> for unknown words and low frequency words

    tag2idx = {tag: idx + 1 for idx, tag in enumerate(train_tags)}
    tag2idx['<pad>'] = 0    # use <pad> for padding tag

    chars = {char for words, _ in raw_train for word in words for char in word}
    char2idx = {char: idx + 2 for idx, char in enumerate(chars)}
    char2idx['<pad>'] = 0
    char2idx['<unk>'] = 1

    return word2idx, tag2idx, char2idx

def load_glove(glove_path):
    with open(glove_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    glove_embeddings = {}
    for line in lines:
        parts = line.strip().split()
        word = parts[0]
        vector = torch.tensor([float(v) for v in parts[1:]])
        glove_embeddings[word] = vector
    return glove_embeddings

def generate_embedding_matrix(word2idx, embedding_dim, glove_embeddings):
    vocab_size = len(word2idx)
    embedding_mat = torch.zeros(vocab_size, embedding_dim)
    mean_glove = torch.mean(torch.stack(list(glove_embeddings.values())), dim=0)
    for word, idx in word2idx.items():
        if word in glove_embeddings:
            embedding_mat[idx] = glove_embeddings[word]
        elif word.lower() in glove_embeddings:
            embedding_mat[idx] = glove_embeddings[word.lower()] + torch.randn(embedding_dim)
        else:
            embedding_mat[idx] = mean_glove + torch.randn(embedding_dim)
            # embedding_mat[idx] = torch.randn(embedding_dim)
    embedding_mat[0] = torch.zeros(embedding_dim)