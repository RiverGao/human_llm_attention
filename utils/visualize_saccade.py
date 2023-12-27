import numpy as np
import pickle
import matplotlib
from matplotlib import pyplot as plt

dataset = 'GeCo'
view = 'num'
p = 1
residue = False

problems = {1: [0, 20], 2: [10, 27, 41, 42, 44, 47, 54]}  # problematic subject indices
file_prefix = 'residue_' if residue else ''
font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 18}
matplotlib.rc('font', **font)

with open(f'../golden_attention/{dataset}/{file_prefix}saccade_{view}_p{p}.pkl', 'rb') as f:
    saccade_data = pickle.load(f)  # subj, article, sentence, (word, word)

with open(f'../sentence_data/{dataset}/sentences_p1.txt', 'r') as f:
    # p2 sentences are wrongly listed
    articles = f.read().strip().split('\n\n')  # articles are separated by two newlines

n_subj = 52 if p == 1 else 56
n_arti = 5
for arti in range(n_arti):
    # iterate over each sentence in each article
    sentences = articles[arti].strip().split('\n')  # sentences in one article are separated by one newline
    for k_sent, s_sent in enumerate(sentences):
        sent_words = s_sent.strip().split()
        saccade_arrays = []
        for subj in range(n_subj):
            # collect mean saccade of all subjects
            if subj in problems[p]:
                # print(f'Problematic subject {subj} in L{p}')
                continue
            try:
                indiv_saccade = saccade_data[subj][arti][k_sent]  # (n_word, n_word) array
                if indiv_saccade.shape[0] == len(sent_words):
                    saccade_arrays.append(indiv_saccade)
                    break
                else:
                    print(f'L{p}A{arti}S{k_sent}, Sub{subj} has sentence length {indiv_saccade.shape[0]} instead of {len(sent_words)}')
            except IndexError:
                pass

        mean_saccade = np.mean(saccade_arrays, axis=0)  # (n_word, n_word) array

        fig, ax = plt.subplots(figsize=(10, 10))
        # saccade_graph = ax.matshow(np.tril(mean_saccade))
        saccade_graph = ax.matshow(mean_saccade)
        ax.set_xticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=30, ha='left')
        ax.set_yticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=60, ha='right')
        ax.set_xlabel('Word To')
        ax.set_ylabel('Word From')
        ax.set_title(f'L{p}, Subject {subj}, Article {arti}, Sentence {k_sent}, {"Non-Trivial" if residue else ""} Saccade {view.capitalize()}')
        fig.colorbar(saccade_graph, ax=ax)
        fig.tight_layout()
        fig.savefig(f'../results/figs/reading_brain/human/{view}/L{p}/{file_prefix}a{arti}s{k_sent}.png', dpi=100)
        plt.close(fig)
        break
    break
