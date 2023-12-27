import numpy as np
import sys
import pickle


model_name = sys.argv[1]
model_size = sys.argv[2]
instr_prefix = sys.argv[3] + '_' if len(sys.argv) > 3 else ''

# model_name = 'llama'
# model_size = '7B'
# instr_prefix = ''

d_size_layer = {
    'gpt2-base': 12,
    'gpt2-large': 36,
    'gpt2-multi': 24,
    'bert-base': 12,
    'bert-large': 24,
    't5-base': 12,
    't5-large': 24,
    'llama-7B': 32,
    'llama-13B': 40,
    'llama-30B': 60,
    'llama-65B': 80
}
if model_name in ['llama', 'alpaca', 'vicuna', 'vicuna-old']:
    n_layers = d_size_layer['llama-' + model_size]
else:
    n_layers = d_size_layer[model_name + '-' + model_size]

sentence_lengths = [
    [7, 10, 11, 11, 13, 11, 10, 10, 10, 9, 9, 15, 12, 10, 11, 7, 7, 9, 10, 9, 10, 8, 10, 9, 11, 10, 10, 10, 8, 10, 13],
    [10, 10, 10, 10, 10, 10, 10, 11, 10, 9, 9, 12, 9, 12, 11, 8, 8, 9, 11, 10, 9, 9, 8, 10, 9, 11, 8, 10, 9, 10, 10],
    [8, 8, 13, 13, 10, 14, 7, 11, 16, 9, 9, 11, 12, 13, 9, 10, 9, 12, 12, 13, 13, 10, 12, 11, 11, 9, 13, 8],
    [11, 11, 10, 8, 10, 9, 9, 11, 12, 13, 14, 12, 13, 11, 9, 11, 14, 13, 9, 12, 12, 8, 10, 10, 13, 11, 11, 10],
    [9, 9, 12, 13, 13, 13, 12, 9, 8, 12, 12, 12, 9, 10, 9, 11, 15, 11, 7, 9, 9, 9, 10, 13, 8, 6, 10, 9, 9, 6]
]


def preprocess(layer_attn, n_words):
    # input shape: (n_head, max_sent_len, max_sent_len)
    mean_att_mat = layer_attn.mean(axis=0)
    att_mat = mean_att_mat[:n_words, :n_words]  # shape: (n_words, n_words)
    residual_att = np.eye(att_mat.shape[0])
    att_mat += residual_att
    att_mat /= att_mat.sum(axis=-1)[..., None]
    return mean_att_mat


# attention rollout is to multiply layer-wise attentions
# input shape:  (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
# output shape: (n_arti, max_n_sents, 1,      max_sent_len, max_sent_len) for each layer
joint_attentions = None  # caching previous rollouts
for layer in range(n_layers):
    model_layer_attn = np.load(
        f'../model_attention/reading_brain/{model_name}/{model_size}/p1/{instr_prefix}rb_p1_layer{layer}.npy'
    )  # (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
    if joint_attentions is None:
        shape = list(model_layer_attn.shape)
        shape[2] = 1
        joint_attentions = np.zeros(shape)  # only heads are averaged

    for arti in range(model_layer_attn.shape[0]):
        n_sents = len(sentence_lengths[arti])
        for sentj in range(n_sents):
            n_words = sentence_lengths[arti][sentj]

            raw_attn = model_layer_attn[arti, sentj]
            aug_attn = preprocess(raw_attn, n_words)

            if layer == 0:
                joint_attentions[arti, sentj, 0] = aug_attn
            else:
                joint_attentions[arti, sentj, 0] = aug_attn @ joint_attentions[arti, sentj, 0]

    np.save(f'../model_attention/reading_brain/{model_name}/{model_size}/rollout/{instr_prefix}rb_p1_layer{layer}.npy', joint_attentions)
