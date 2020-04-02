# coding=utf-8


import os
import numpy as np


def load_data(data_dir, split=None):
    data = []
    if split is None:
        files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.txt')]
    else:
        if not split.endswith('.txt'):
            split += '.txt'
        files = [os.path.join(data_dir, f'{split}')]
    for file in files:
        with open(file) as f:
            for line in f:
                text1, text2, label = line.rstrip().split('\t')
                data.append({
                    'text1': text1,
                    'text2': text2,
                    'target': label,
                })
    return data


def load_embeddings(file, vocab, dim, lower, mode='freq'):
    embedding = np.zeros((len(vocab), dim))
    count = np.zeros((len(vocab), 1))
    with open(file) as f:
        for line in f:
            elems = line.rstrip().split()
            if len(elems) != dim + 1:
                continue
            token = elems[0]
            if lower and mode != 'strict':
                token = token.lower()
            if token in vocab:
                index = vocab.index(token)
                vector = [float(x) for x in elems[1:]]
                if mode == 'freq' or mode == 'strict':
                    if not count[index]:
                        embedding[index] = vector
                        count[index] = 1.
                elif mode == 'last':
                    embedding[index] = vector
                    count[index] = 1.
                elif mode == 'avg':
                    embedding[index] += vector
                    count[index] += 1.
                else:
                    raise NotImplementedError('Unknown embedding loading mode: ' + mode)
    if mode == 'avg':
        inverse_mask = np.where(count == 0, 1., 0.)
        embedding /= count + inverse_mask
    return embedding.tolist()


