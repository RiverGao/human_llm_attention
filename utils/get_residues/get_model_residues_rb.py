import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import sys


def tril_idx(n):
    # 00, 01, 11, 02, 12, 22
    x = []
    y = []
    for i in range(n):
        for j in range(n):
            if j > i:
                break
            x.append(i)
            y.append(j)
    return np.array(x), np.array(y)


def apply_values(values, target, size):
    # size is the width (also height) of the original value matrix, i.e. n_words
    it = iter(values)
    for i in range(size):
        for j in range(size):
            if j > i:
                break
            target[i, j] += next(it)


model_name = 'vicuna'
model_size = '13B'
attn_method = 'rollout'

# model_name = sys.argv[1]
# model_size = sys.argv[2]

label_types = ['self', 'prev', 'start']
d_size_layers = {'large': 36, '7B': 32, '13B': 40, '7Bhf': 32, '13Bhf': 40, '30B': 60}
d_size_heads = {'large': 20, '7B': 32, '13B': 40, '7Bhf': 32, '13Bhf': 40, '30B': 52}
n_layers = d_size_layers[model_size]
n_heads = 1 if attn_method else d_size_heads[model_size]
np.random.seed(42)

label_data = []  # type, article, sentence, word, word
for label_type in label_types:
    with open(f'../../golden_attention/reading_brain/label_{label_type}.pkl', 'rb') as f:
        label_data.append(pickle.load(f))  # article, sentence, word, word

sentence_lengths = [[] for i in range(5)]  # store the length of each sentence, list: article, sentence
r2_scores = []  # R^2 of LR in each layer

for layer in range(n_layers):
    # do linear regression in each layer independently
    # output residues of each layer
    flatten_label_data = []  # length: 2-D, n_articles * n_sentences * n_words * (n_words + 1) / 2, n_labels
    flatten_model_attn = []  # length: 2-D, n_articles * n_sentences * n_words * (n_words + 1) / 2, n_heads

    # read model attentions in this layer
    model_layer_attn = np.load(
        f'../../model_attention/reading_brain/{model_name}/{model_size}/{attn_method if attn_method else "p1"}/rb_p1_layer{layer}.npy'
    )  # (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
    assert not np.isnan(np.sum(model_layer_attn)), f'Layer {layer} has NaN values'

    # flatten labels as well as model attentions
    # store lengths of all sentences for recovery
    for arti in range(5):
        sentence_data0_in_article = label_data[0][arti]  # n_sent, n_words, n_words
        for sentj in range(len(sentence_data0_in_article)):
            sentence_data0 = np.array(sentence_data0_in_article[sentj])  # n_words, n_words
            n_words = len(sentence_data0)
            if layer == 0:
                # only store sentence lengths once, and reuse it afterwards
                sentence_lengths[arti].append(n_words)

            # flattened indices
            tril_x, tril_y = tril_idx(n_words)

            # flatten model attention
            model_layer_arti_sentj = model_layer_attn[arti][sentj]  # n_head, max_sent_len, max_sent_len
            flat_attn = model_layer_arti_sentj[:, tril_x, tril_y].T.tolist()  # n_words * (n_words + 1) / 2, n_head
            flatten_model_attn.extend(flat_attn)  # as y in LR, n_arti * n_sent * n_words * (n_words + 1) / 2, n_head

            # flatten labels
            flat_lbl_for_types = []  # n_labels, each element is array(n_words * (n_words + 1) / 2)
            for typek in range(len(label_types)):
                sentence_data = np.array(label_data[typek][arti][sentj])  # array(n_words, n_words)
                flat_lbl_for_types.append(sentence_data[tril_x, tril_y])  # append an array(n_words * (n_words + 1) / 2)
            flat_lbl = np.stack(flat_lbl_for_types).T.tolist()  # n_words * (n_words + 1) / 2, n_labels
            flatten_label_data.extend(flat_lbl)

    # recover attention structure by adding sentence attentions on the zeros_like matrix
    residue_layer_attn = np.zeros_like(model_layer_attn)  # (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)

    # do linear regression for each attention head
    r2_scores_heads = []
    for head in range(n_heads):
        X = flatten_label_data
        y = np.array(flatten_model_attn)[:, head]
        reg_head = LinearRegression().fit(X, y)
        score = reg_head.score(X, y)
        r2_scores_heads.append(score)
        # print(f'Label Linear Regression for Layer {layer} Head {head}\nScore: {score}\nCoef: {reg_head.coef_}, Intercept: {reg_head.intercept_}')

        # get residue
        flatten_head_prediction = reg_head.predict(flatten_label_data)  # array(n_arti * n_sent * n_words * (n_words + 1) / 2)
        flatten_head_residue = y - flatten_head_prediction  # array(n_arti * n_sent * n_words * (n_words + 1) / 2)
        start_pos = 0  # pointer for copying residue values

        # recover residue to the original format: # (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
        for arti in range(5):
            for sentj in range(len(sentence_lengths[arti])):
                n_words = sentence_lengths[arti][sentj]  # number of words in this sentence
                n_values = n_words * (n_words + 1) // 2  # number of values in the lower triangle matrix
                values_to_apply = flatten_head_residue[start_pos: start_pos + n_values]
                apply_values(values_to_apply, residue_layer_attn[arti, sentj, head], n_words)
                start_pos += n_values

    r2_scores.append(np.mean(r2_scores_heads))
    np.save(f'../../model_attention/reading_brain/{model_name}/{model_size}/{attn_method if attn_method else "p1"}/residue_rb_p1_layer{layer}.npy', residue_layer_attn)

print(np.mean(r2_scores))
np.save(f'../../results/lr_scores/{model_name}/{model_size}/{attn_method + "_" if attn_method else ""}scores_rb_p1.npy', np.array(r2_scores))
