import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_distances


def show_raw_distrib(data, fst_emb):
    d = np.array([cosine_distances([fst_emb[w0]], [fst_emb[w1]])[0][0] for w0, w1 in data[:, :2]])
    plt.hist(d[data[:, 2] == 'S'], label='synonyms')
    plt.hist(d[data[:, 2] == 'A'], label='antonyms')
    plt.legend()


def show_atribute(data, embedding):
    plt.title('Fasttext train cosine distance distribution')
    plt.ylabel('Number of pairs')
    plt.xlabel('Cosine distance')
    show_raw_distrib(data, embedding)
    plt.show()
