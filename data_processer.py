import pickle
import random
import numpy as np
import pandas as pd
import jsonlines
from tqdm import tqdm

from transformers import pipeline

import nltk

import torch
from torch.utils.data import Dataset


# iter_train = np.array(pd.read_csv('input_data/train65k.csv'))
# iter_test = np.array(pd.read_csv('input_data/test14k.csv'))

class read_files:
    def __init__(self, path):
        self.path = path

    def read_json(self):
        dataset = []
        with open(self, "r+", encoding='utf-8') as f:
            for item in tqdm(jsonlines.Reader(f), desc='reading json file'):
                pairs = [item['text'].replace('\n', ' '), int(item['stars'])]
                dataset.append(pairs)
        print('-' * 40 + '\nreading json file finished\n')

        return dataset

    def read_csv(self):
        dataset = np.array((pd.read_csv(self)))
        print('-' * 40 + '\nreading csv file finished\n')

        return dataset

    def read_pkl(self):
        with open(self, "rb") as f:
            embeddings = pickle.load(f)
        print('-' * 40 + '\nreading pkl file finished\n')

        return embeddings


class TripletDataset(Dataset):
    def __init__(self, pairs_dataset, emb, is_triplet=True, stack_samples=True, return_words=False):
        self.emb = emb
        self.stack_samples = stack_samples
        self.dataset = pairs_dataset
        self.is_triplet = is_triplet
        self.relation_encoder = {'S': 0, 'A': 1}
        self._len = len(self.dataset)
        self.return_words = return_words
        if self.is_triplet:
            self.word_statistic = pd.DataFrame(self.pair_statistic()).T
            self.words = self.get_triplet_words(self.word_statistic)
            self._len = len(self.words)

    def __getitem__(self, index):
        if self.is_triplet:
            word = self.words[index]
            anchor = self.emb[word]
            ant, syn = self._get_word_pairs(index)
            ant = self._rebuild_pairs(word, ant)
            syn = self._rebuild_pairs(word, syn)
            positive = self.emb[random.choice(syn)]
            negative = self.emb[random.choice(ant)]
            return (anchor, positive, negative), []

        if not self.is_triplet:
            word = self.dataset[index][0]
            anchor = self.emb[word]
            related_word = self.dataset[index][1]
            related_word_emb = self.emb[related_word]
            relation = self.relation_encoder[self.dataset[index][2]]
            if self.stack_samples:
                x = np.hstack((anchor, related_word_emb))
                return x, relation
            else:
                if self.return_words:
                    return (anchor, related_word_emb), relation, (word, related_word)
                else:
                    return (anchor, related_word_emb), relation
            # return (anchor, related_word_emb), relation

    def _get_word_pairs(self, idx):
        d = np.where(self.dataset[:, [0, 1]] == self.words[idx])[0]
        split = self.dataset[d]
        ant = split[split[:, 2] == 'A']
        syn = split[split[:, 2] == 'S']
        return ant, syn

    def _rebuild_pairs(self, word, array):
        return list(set([item for sublist in array[:, [0, 1]] for item in sublist]) - {word})

    def _append_word_to_dict(self, d, word):
        d[word] = {'S': 0, 'A': 0}

    def _check_word(self, d, w):
        if w not in d.keys():
            self._append_word_to_dict(d, w)

    def pair_statistic(self):
        result_dict = dict()
        for w0, w1, r in tqdm(self.dataset[:, [0, 1, 2]]):
            self._check_word(result_dict, w0)
            self._check_word(result_dict, w1)

            result_dict[w0][r] += 1
            result_dict[w1][r] += 1
        return result_dict

    def get_triplet_words(self, df):
        return df[((df['A'] > 0) & (df['S'] > 0))].index.to_list()

    def __len__(self):
        return self._len


def get_BERT_embedding(word_bag, path):
    bert_embedding = pipeline('feature-extraction')
    embedding_dict = {}
    for word in tqdm(word_bag, desc='BERT embedding'):
        embedding = bert_embedding(word)[0][0]
        embedding_dict[word] = np.array(embedding, dtype='float32')
    print('-' * 40 + '\nget embedding finished\n')

    with open(path, "wb") as fp:
        pickle.dump(embedding_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print('saved path: ' + path)

    return embedding_dict


def get_wordbag(dataset, data_type):
    wordbag = []
    if data_type == 'as_list':
        for pair in tqdm(dataset, desc='making wordbag - as_list'):
            if pair[0] not in wordbag:
                wordbag.append(pair[0])
            if pair[1] not in wordbag:
                wordbag.append(pair[1])

    elif data_type == 'input':
        for pair in tqdm(dataset, desc='making wordbag - input'):
            if nltk.word_tokenize(pair[0]) not in wordbag:
                wordbag.extend(nltk.word_tokenize(pair[0]))
    print('-' * 40 + '\nget wordbag done\n')

    return wordbag


def get_intersection(corpus_words, as_list):
    anto_syno = []
    for pair in tqdm(as_list, desc='intersection'):
        if pair[0] in corpus_words and pair[1] in corpus_words:
            anto_syno.append(pair)
    anto_syno = np.array(anto_syno)
    print('-' * 40 + '\nget intersection done\n')

    return anto_syno


def get_train_val_val_test(train, test, emb):
    train_triplet_dataloader = torch.utils.data.DataLoader(TripletDataset(train, emb,
                                                                          is_triplet=True),
                                                           batch_size=64,
                                                           shuffle=False,
                                                           num_workers=0)

    val_triplet_dataloader = torch.utils.data.DataLoader(TripletDataset(test, emb,
                                                                        is_triplet=True),
                                                         batch_size=64,
                                                         shuffle=False,
                                                         num_workers=0)

    val_pair_dataloader = torch.utils.data.DataLoader(TripletDataset(test, emb,
                                                                     is_triplet=False,
                                                                     stack_samples=False),
                                                      batch_size=64,
                                                      shuffle=False,
                                                      num_workers=0)

    test_dataset = TripletDataset(test, emb, is_triplet=False, stack_samples=False)

    return train_triplet_dataloader, val_triplet_dataloader, val_pair_dataloader, test_dataset
