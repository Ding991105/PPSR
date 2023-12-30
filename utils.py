import torch
import numpy as np


# news+topic, news+subtopic, all_news, subtopic+topic, all_subtopic
def construct_graph(history_topic_padding, history_subtopic_padding):
    batch_size, seq_len = history_topic_padding.shape
    num_t_nodes = [len(set(h)) for h in history_topic_padding]
    num_s_nodes = [len(set(h)) for h in history_subtopic_padding]
    n_t, n_s = max(num_t_nodes), max(num_s_nodes)

    alias_inputs_topic, alias_inputs_subtopic = [], []
    topics, subtopics = [], []
    adj = np.zeros((batch_size, 2 * n_t + n_s + 2, seq_len + n_t + n_s))  # bs, n_edge, n_node

    for bs in range(batch_size):
        u_input_topic = history_topic_padding[bs]
        u_input_subtopic = history_subtopic_padding[bs]

        topic_node = np.unique(u_input_topic)
        subtopic_node = np.unique(u_input_subtopic)
        alias_inputs_topic.append([np.where(topic_node == i)[0][0] for i in u_input_topic])
        alias_inputs_subtopic.append([np.where(subtopic_node == i)[0][0] for i in u_input_subtopic])
        topics.append(topic_node.tolist() + (n_t - len(topic_node)) * [0])
        subtopics.append(subtopic_node.tolist() + (n_s - len(subtopic_node)) * [0])

        for i in range(len(u_input_topic)):
            if u_input_topic[i] == 0:
                break
            u_t = np.where(topic_node == u_input_topic[i])[0][0]
            u_s = np.where(subtopic_node == u_input_subtopic[i])[0][0]

            # hyperedge news+topic
            adj[bs][u_t][i] = 1
            adj[bs][u_t][u_t + seq_len] = 1

            # hyperedge news+subtopic
            adj[bs][u_s + n_t][i] = 1
            adj[bs][u_s + n_t][u_s + seq_len + n_t] = 1

            # hyperedge all_news
            adj[bs][-1][i] = 1

            # hyperedge subtopic+topic
            adj[bs][u_t + n_t + n_s][u_s + seq_len + n_t] = 1
            adj[bs][u_t + n_t + n_s][u_t + seq_len] = 1

            # hyperedge all_subtopic
            adj[bs][-2][u_s + seq_len + n_t] = 1

    alias_inputs_topic = np.asarray(alias_inputs_topic)
    alias_inputs_subtopic = np.asarray(alias_inputs_subtopic)
    topics = np.asarray(topics)
    subtopics = np.asarray(subtopics)
    return [alias_inputs_topic, alias_inputs_subtopic, topics, subtopics, adj]

