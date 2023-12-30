import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from aggregator import *


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, d_k: int, d_v: int):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = self.h * self.d_v
        self.attention_scalar = math.sqrt(float(self.d_k))
        self.W_K = nn.Linear(d_model, self.h*self.d_k, bias=False)
        self.W_Q = nn.Linear(d_model, self.h*self.d_k, bias=True)
        self.W_V = nn.Linear(d_model, self.h*self.d_v, bias=True)

    def initialize(self):
        nn.init.zeros_(self.W_Q.bias)
        nn.init.zeros_(self.W_V.bias)

    def forward(self, Q, K, V, mask):
        batch_size = Q.size(0)
        len_k = K.size(1)
        batch_h_size = batch_size * self.h
        Q = self.W_Q(Q).view([batch_size, -1, self.h, self.d_k])                    # [batch_size, len_q, h, d_k]
        K = self.W_K(K).view([batch_size, -1, self.h, self.d_k])                    # [batch_size, len_k, h, d_k]
        V = self.W_V(V).view([batch_size, -1, self.h, self.d_v])                    # [batch_size, len_k, h, d_v]
        Q = Q.transpose(1, 2).contiguous().view([batch_h_size, -1, self.d_k])       # [batch_size * h, len_q, d_k]
        K = K.transpose(1, 2).contiguous().view([batch_h_size, -1, self.d_k])       # [batch_size * h, len_k, d_k]
        V = V.transpose(1, 2).contiguous().view([batch_h_size, -1, self.d_v])       # [batch_size * h, len_k, d_v]
        A = torch.bmm(Q, K.transpose(1, 2).contiguous()) / self.attention_scalar    # [batch_size * h, len_q, len_k]
        if mask is not None:                                                        # [batch_size, len_q]
            mask = mask.repeat(self.h, len_k).view(batch_h_size, -1, len_k)         # [batch_size * h, len_q, len_k]
            alpha = F.softmax(A.masked_fill(mask == 0, -1e9), dim=2)
        else:
            alpha = F.softmax(A, dim=2)                                             # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view([batch_size, self.h, -1, self.d_v])          # [batch_size, h, len_q, d_v]
        out = out.transpose(1, 2).contiguous().view([batch_size, -1, self.out_dim]) # [batch_size, len_q, h * d_v]
        return out


class Attention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        super(Attention, self).__init__()
        self.affine1 = nn.Linear(feature_dim, attention_dim, bias=True)
        self.affine2 = nn.Linear(attention_dim, 1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, feature, mask):
        attention = torch.tanh(self.affine1(feature))                                 # [batch_size, length, attention_dim]
        a = self.affine2(attention).squeeze(dim=2)                                    # [batch_size, length]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1) # [batch_size, 1, length]
        else:
            alpha = F.softmax(a, dim=1).unsqueeze(dim=1)                              # [batch_size, 1, length]
        out = torch.bmm(alpha, feature).squeeze(dim=1)                                # [batch_size, feature_dim]
        return out


class NewsEncoder(nn.Module):
    def __init__(self, opt, word_embeddings):
        super(NewsEncoder, self).__init__()
        self.word_embeddings = nn.Embedding(num_embeddings=word_embeddings.shape[0], embedding_dim=opt.Glove_dim)
        self.word_embeddings.weight.data.copy_(torch.FloatTensor(word_embeddings))
        self.MHA = MultiHeadAttention(opt.num_head_n, opt.Glove_dim, opt.head_dim, opt.head_dim)
        self.attention = Attention(opt.num_head_n * opt.head_dim, opt.num_head_n * opt.head_dim)
        self.dropout = nn.Dropout(p=opt.dropout)

    def initialize(self):
        self.MHA.initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask):
        title_embeddings = self.dropout(self.word_embeddings(title_text))  # N, W, d
        h = F.relu(self.MHA(title_embeddings, title_embeddings, title_embeddings, mask=title_mask))  # N, W, h * d_v
        news_representation = self.attention(h, mask=title_mask)
        return news_representation


class UserEncoder(nn.Module):
    def __init__(self, opt):
        super(UserEncoder, self).__init__()
        self.MHA = MultiHeadAttention(opt.num_head_u, opt.num_head_u * opt.head_dim, opt.head_dim, opt.head_dim)
        self.attention = Attention(opt.num_head_u * opt.head_dim, opt.num_head_u * opt.head_dim)
        self.dropout = nn.Dropout(p=opt.dropout)

    def initialize(self):
        self.MHA.initialize()
        self.attention.initialize()

    def forward(self, hidden_history, history_mask):
        h = F.relu(self.dropout(self.MHA(hidden_history, hidden_history, hidden_history, mask=history_mask)))  # bs, L, h * d_v
        user_representation = self.attention(h, mask=history_mask)
        return user_representation


class NeighbourEncoder(nn.Module):
    def __init__(self, opt):
        super(NeighbourEncoder, self).__init__()
        self.MHA = MultiHeadAttention(opt.num_head_c, opt.num_head_c * opt.head_dim, opt.head_dim, opt.head_dim)

    def initialize(self):
        self.MHA.initialize()

    def forward(self, hidden_candidate, hidden_neighbour):
        hidden_candidate = torch.unsqueeze(hidden_candidate, dim=1)  # bs*K, 1, h * d_v + 2dim
        hidden_neighbour = torch.cat((hidden_neighbour, hidden_candidate), dim=1)  # bs*K, n+1, h * d_v + 2dim
        candidate_representation = self.MHA(hidden_candidate, hidden_neighbour, hidden_neighbour, mask=None)  # bs*K, 1, h * d_v
        return candidate_representation


class GateNetwork(nn.Module):
    def __init__(self, d_model):
        super(GateNetwork, self).__init__()
        self.dense = nn.Linear(d_model, 1, bias=True)

    def forward(self, gate, scores1, scores2):
        w = torch.sigmoid(self.dense(gate))  # bs * 1
        output = w * scores1 + (1 - w) * scores2
        return output


class P_SHE(nn.Module):
    def __init__(self, device, opt, word_embed_matrix, n_topic, n_subtopic, n_pop):
        super(P_SHE, self).__init__()
        self.device = device
        self.loss_function = nn.CrossEntropyLoss()

        self.news_encoder = NewsEncoder(opt, word_embed_matrix)
        self.user_encoder = UserEncoder(opt)
        self.neighbour_encoder = NeighbourEncoder(opt)

        self.topicID_dim = opt.topicID_dim
        self.subtopicID_dim = opt.subtopicID_dim
        self.news_dim = opt.num_head_n * opt.head_dim
        self.popularity_dim = opt.popularity_dim

        self.topic_embedding = nn.Embedding(n_topic, self.topicID_dim)
        self.subtopic_embedding = nn.Embedding(n_subtopic, self.subtopicID_dim)
        self.news_clickNum_embedding = nn.Embedding(n_pop, self.popularity_dim)

        self.dense1 = nn.Linear(self.topicID_dim, self.topicID_dim, bias=True)
        self.dense2 = nn.Linear(self.subtopicID_dim, self.subtopicID_dim, bias=True)
        self.dense3 = nn.Linear(self.topicID_dim, self.news_dim, bias=True)
        self.dense4 = nn.Linear(self.subtopicID_dim, self.news_dim, bias=True)
        self.dense5 = nn.Linear(self.news_dim, self.topicID_dim, bias=True)
        self.dense6 = nn.Linear(self.news_dim, self.subtopicID_dim, bias=True)
        self.dense7 = nn.Linear(self.popularity_dim, self.popularity_dim, bias=True)
        self.dense8 = nn.Linear(self.popularity_dim, 1, bias=True)

        self.HHGAT = HHGAT(opt.gnn_dim, opt.gnn_dim, opt.gnn_dim, opt.step, dropout=opt.dropout)
        self.gate = GateNetwork(opt.num_head_u * opt.head_dim)

    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()
        self.neighbour_encoder.initialize()

    def forward(self, history_mask, candidate_padding,
                history_title_word, history_title_word_mask,
                candidate_title_word, candidate_title_word_mask,
                neighbour_title_word, neighbour_title_word_mask,
                candidate_topic, neighbour_topic,
                candidate_subtopic, neighbour_subtopic,
                candidate_clickNum_label,
                alias_inputs_topic, alias_inputs_subtopic, topics, subtopics, adj):
        bs = history_title_word.shape[0]
        L = history_title_word.shape[1]
        W = history_title_word.shape[2]
        K = candidate_title_word.shape[1]
        n = neighbour_title_word.shape[2]

        topic_embedding = F.relu(self.dense1(self.topic_embedding.weight))
        subtopic_embedding = F.relu(self.dense2(self.subtopic_embedding.weight))
        news_clickNum_embedding = F.relu((self.dense7(self.news_clickNum_embedding.weight)))

        history_topic_emb = self.dense3(topic_embedding[topics])
        candidate_topic_emb = topic_embedding[candidate_topic]
        neighbour_topic_emb = topic_embedding[neighbour_topic]
        history_subtopic_emb = self.dense4(subtopic_embedding[subtopics])
        candidate_subtopic_emb = subtopic_embedding[candidate_subtopic]
        neighbour_subtopic_emb = subtopic_embedding[neighbour_subtopic]

        candidate_clickNum_emb = news_clickNum_embedding[candidate_clickNum_label]

        history_title_word = history_title_word.view(bs * L, W)
        history_title_word_mask = history_title_word_mask.view(bs * L, W)
        candidate_title_word = candidate_title_word.view(bs * K, W)
        candidate_title_word_mask = candidate_title_word_mask.view(bs * K, W)
        neighbour_title_word = neighbour_title_word.view(bs * K * n, W)
        neighbour_title_word_mask = neighbour_title_word_mask.view(bs * K * n, W)
        history_news_title_embeddings = self.news_encoder(history_title_word, history_title_word_mask)
        candidate_news_title_embeddings = self.news_encoder(candidate_title_word, candidate_title_word_mask)
        neighbour_news_title_embeddings = self.news_encoder(neighbour_title_word, neighbour_title_word_mask)
        history_news_title_embeddings = history_news_title_embeddings.view(bs, L, -1)
        candidate_news_title_embeddings = candidate_news_title_embeddings.view(bs, K, -1)
        neighbour_news_title_embeddings = neighbour_news_title_embeddings.view(bs, K, n, -1)

        x = torch.cat((history_news_title_embeddings, history_topic_emb, history_subtopic_emb), dim=1)
        x = self.HHGAT(x, adj)
        history_news_title_embeddings, history_topic_emb, history_subtopic_emb = \
            x.split((history_news_title_embeddings.shape[1], history_topic_emb.shape[1], history_subtopic_emb.shape[1]), dim=1)
        get = lambda index: history_subtopic_emb[index][alias_inputs_subtopic[index]]
        history_subtopic_emb = torch.stack([get(i) for i in torch.arange(len(alias_inputs_subtopic)).long()])
        get = lambda index: history_topic_emb[index][alias_inputs_topic[index]]
        history_topic_emb = torch.stack([get(i) for i in torch.arange(len(alias_inputs_topic)).long()])
        history_topic_emb = self.dense5(history_topic_emb)
        history_subtopic_emb = self.dense6(history_subtopic_emb)

        hidden_history = torch.cat((history_news_title_embeddings, history_topic_emb, history_subtopic_emb), dim=-1)
        hidden_candidate = torch.cat((candidate_news_title_embeddings, candidate_topic_emb, candidate_subtopic_emb), dim=-1)
        hidden_neighbour = torch.cat((neighbour_news_title_embeddings, neighbour_topic_emb, neighbour_subtopic_emb), dim=-1)

        hidden_candidate_enhanced = self.neighbour_encoder(hidden_candidate.view(bs * K, -1), hidden_neighbour.view(bs * K, n, -1)).view(bs, K, -1)
        user_rep = self.user_encoder(hidden_history, history_mask)  # bs, h * d_v

        scores = torch.matmul(user_rep.unsqueeze(1), hidden_candidate_enhanced.transpose(1, 2)).squeeze(1)  # bs, K
        scores_pop = self.dense8(candidate_clickNum_emb).squeeze(dim=-1)
        candidate_mask_inf = torch.where(candidate_padding == 0, float('-inf') * torch.ones_like(candidate_padding), torch.zeros_like(candidate_padding).float())
        final_scores = self.gate(user_rep, scores, scores_pop) + candidate_mask_inf

        candidate_mask = torch.where(candidate_padding == 0, torch.zeros_like(candidate_padding), torch.ones_like(candidate_padding))
        candidate_lengths = torch.sum(candidate_mask, dim=-1)

        return final_scores, candidate_lengths
