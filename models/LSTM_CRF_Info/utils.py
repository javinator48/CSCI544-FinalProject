import torch


LABEL = ["O", "I-ORG", "B-PER", "I-LOC", "B-LOC", "I-PER", "B-MISC", "B-ORG", "I-MISC"]


def read_data(path, is_train=True):
    sentences, ner_tags = [], []
    sentence, ner_tag = [], []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                sentences.append(sentence)
                ner_tags.append(ner_tag)
                break
            line = line[:-1]
            if line == "":
                sentences.append(sentence)
                ner_tags.append(ner_tag)
                sentence = []
                ner_tag = []
            elif is_train:
                _, s, n = line.split(' ')
                sentence.append(s)
                ner_tag.append(n)
            else:
                _, s = line.split(' ')
                sentence.append(s)
    return sentences, ner_tags
    
    
def get_word2idx(sentences, min_count=1):
    word_count = {}
    for sentence in sentences:
        for w in sentence:
            if w not in word_count:
                word_count[w] = 0
            word_count[w] += 1
    word2idx = {}
    for w in word_count:
        if word_count[w] >= min_count:
            word2idx[w] = len(word2idx) + 2
    return word2idx


def get_glove_embedding(sentence, glove_dict):
    result = torch.zeros((0, 102))
    for w in sentence:
        embed = glove_dict[w.lower()] if w.lower() in glove_dict else torch.zeros(100)
        embed = torch.cat([
            embed,
            torch.tensor(float(any(w.isupper() for c in w))).unsqueeze(0),
            torch.tensor(1.).unsqueeze(0)])
        result = torch.cat([result, embed.unsqueeze(0)], dim=0)
    return result
            