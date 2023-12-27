import numpy as np
import sys
import networkx as nx


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


def get_adjmat(mat):
    # input shape: (n_layers, n_words, n_words)
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers + 1) * length, (n_layers + 1) * length))
    labels_to_index = {}
    for k in np.arange(length):
        labels_to_index["L0_" + str(k)] = k

    for i in np.arange(1, n_layers + 1):
        for k_f in np.arange(length):
            index_from = i * length + k_f
            label = "L" + str(i) + "_" + str(k_f)
            labels_to_index[label] = index_from
            for k_t in np.arange(length):
                index_to = (i - 1) * length + k_t
                adj_mat[index_from][index_to] = mat[i - 1][k_f][k_t]

    return adj_mat, labels_to_index


def draw_attention_graph(adjmat, labels_to_index, n_layers, length):
    A = adjmat
    G = nx.from_numpy_array(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(G, {(i, j): A[i, j]}, 'capacity')

    pos = {}
    label_pos = {}
    for i in np.arange(n_layers + 1):
        for k_f in np.arange(length):
            pos[i * length + k_f] = ((i + 0.5) * 2, length - k_f)
            label_pos[i * length + k_f] = (i * 2, length - k_f)

    index_to_labels = {}
    for key in labels_to_index:
        index_to_labels[labels_to_index[key]] = key.split("_")[-1]
        if labels_to_index[key] >= length:
            index_to_labels[labels_to_index[key]] = ''

    # plt.figure(1,figsize=(20,12))

    nx.draw_networkx_nodes(G, pos, node_color='green', node_size=50)
    nx.draw_networkx_labels(G, pos=label_pos, labels=index_to_labels, font_size=10)

    all_weights = []
    # 4 a. Iterate through the graph nodes to gather all the weights
    for (node1, node2, data) in G.edges(data=True):
        all_weights.append(data['weight'])  # we'll use this when determining edge thickness

    # 4 b. Get unique weights
    unique_weights = list(set(all_weights))

    # 4 c. Plot the edges - one by one!
    for weight in unique_weights:
        # 4 d. Form a filtered list with just the weight you want to draw
        weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in G.edges(data=True) if
                          edge_attr['weight'] == weight]
        # 4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner

        w = weight  # (weight - min(all_weights))/(max(all_weights) - min(all_weights))
        width = w
        nx.draw_networkx_edges(G, pos, edgelist=weighted_edges, width=width, edge_color='darkblue')

    return G


def compute_flows(G, labels_to_index, input_nodes, length):
    number_of_nodes = len(labels_to_index)
    flow_values = np.zeros((number_of_nodes, number_of_nodes))
    for key in labels_to_index:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G, u, v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer * length + v] = flow_value
            flow_values[u] /= flow_values[u].sum()

    return flow_values


def convert_adjmat_tomats(adjmat, n_layers, l):
    mats = np.zeros((n_layers, l, l))

    for i in np.arange(n_layers):
        mats[i] = adjmat[(i + 1) * l:(i + 2) * l, i * l:(i + 1) * l]

    return mats


# attention rollout is to multiply layer-wise attentions
# input shape:  (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
# output shape: (n_arti, max_n_sents, n_layers, 1, max_sent_len, max_sent_len)
joint_attentions = None  # caching previous rollouts
for layer in range(n_layers):
    model_layer_attn = np.load(
        f'../model_attention/reading_brain/{model_name}/{model_size}/p1/{instr_prefix}rb_p1_layer{layer}.npy'
    )  # (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
    if joint_attentions is None:
        shape = list(model_layer_attn.shape)
        shape[2] = 1
        shape.insert(2, n_layers)
        joint_attentions = np.zeros(shape)  # only heads are averaged

    for arti in range(5):
        n_sents = len(sentence_lengths[arti])
        for sentj in range(n_sents):
            n_words = sentence_lengths[arti][sentj]

            raw_attn = model_layer_attn[arti, sentj]
            aug_attn = preprocess(raw_attn, n_words)

            joint_attentions[arti, sentj, layer, 0] = aug_attn

print(joint_attentions.shape)
attention_flows = np.zeros_like(joint_attentions)  # (n_arti, max_n_sents, n_layers, 1, max_sent_len, max_sent_len)
print(attention_flows.shape)

for arti in range(5):
    n_sents = len(sentence_lengths[arti])
    # n_sents = 2
    for sentj in range(n_sents):
        n_words = sentence_lengths[arti][sentj]
        mat = joint_attentions.squeeze()[arti, sentj, :, :n_words, :n_words]
        res_adj_mat, res_labels_to_index = get_adjmat(mat)
        res_G = draw_attention_graph(res_adj_mat, res_labels_to_index, n_layers=n_layers, length=n_words)

        output_nodes = []
        input_nodes = []
        for key in res_labels_to_index:
            # compute flows from the inputs to the current layer
            if f'L{n_layers + 1}_' in key:
                output_nodes.append(key)
            if res_labels_to_index[key] < n_words:
                # this means, the input tokens (layer 0).
                input_nodes.append(key)

        flow_values = compute_flows(res_G, res_labels_to_index, input_nodes, length=n_words)
        print(flow_values.shape)  # (n_nodes, n_nodes)
        flow_att_mat = convert_adjmat_tomats(flow_values, n_layers=n_layers, l=n_words)
        print(flow_att_mat.shape)  # (n_layers, n_words, n_words)
        pad_len = attention_flows.shape[-1] - n_words
        flow_att_mat = np.pad(flow_att_mat, ((0, 0), (0, pad_len), (0, pad_len)))
        print(flow_att_mat.shape)  # (n_layers, max_sent_len, max_sent_len)
        shape = list(flow_att_mat.shape)
        shape.insert(1, 1)
        flow_att_mat = flow_att_mat.reshape(shape)
        print(flow_att_mat.shape)  # (n_layers, 1, max_sent_len, max_sent_len)
        attention_flows[arti, sentj] = flow_att_mat

print('finally...')
# save layer-wise attention flow
for layer in range(n_layers):
    layer_attn_flow = attention_flows[:, :, layer, :, :, :]  # (n_arti, max_n_sents, 1, max_sent_len, max_sent_len)
    np.save(f'../model_attention/reading_brain/{model_name}/{model_size}/flow/{instr_prefix}rb_p1_layer{layer}.npy', layer_attn_flow)
