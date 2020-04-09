# coding=utf-8
import os, json
import numpy as np
from curLine_file import curLine

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


def load_embeddings_Chinese(folder, vocab, dim, lower, mode='freq'): #  中文字向量
    embedding_npy_file = os.path.join(folder, "char2vecChinese_DbqaSmpLog.npy")  # char2vecChinese_DbqaSmpLogZero.npy
    embedding_npy = np.load(embedding_npy_file)
    vocab_file = os.path.join(folder, "char2idChinese_DbqaSmpLog.json")
    with open(vocab_file, "r") as fr:
        vocab_map = json.load(fr)
    # OOV 初始化为零向量
    embedding = np.zeros((len(vocab), dim))
    count = np.zeros((len(vocab), 1))
    for token,index in vocab_map.items():
        if lower and mode != 'strict':
            token = token.lower()
        if token in vocab:
            index = vocab.index(token)
            vector = embedding_npy[index]  # [float(x) for x in elems[1:]]
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
    embedding = embedding.tolist()

    vector = embedding[0]
    print(curLine(), len(vector), "vector:", type(vector), vector)
    return embedding

def load_embeddings_English(file, vocab, dim, lower, mode='freq'): #  glove的英文词向量
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
