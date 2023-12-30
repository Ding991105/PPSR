import os
import argparse
import pickle
import time
import datetime
from model import *
from evaluate import *
import torch
from data import *
import numpy as np
import tqdm
import json
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MIND-small', help='MIND-large')
parser.add_argument('--batchSize_train', type=int, default=32)
parser.add_argument('--batchSize_test', type=int, default=4)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_epoch', type=list, default=[3, 6, 9, 12], help='the epoch which the learning rate decay')
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--Glove_dim', type=int, default=300)
parser.add_argument('--topicID_dim', type=int, default=100)
parser.add_argument('--subtopicID_dim', type=int, default=100)
parser.add_argument('--popularity_dim', type=int, default=100)
parser.add_argument('--gnn_dim', type=int, default=400)
parser.add_argument('--num_head_n', type=int, default=16)
parser.add_argument('--num_head_u', type=int, default=24)
parser.add_argument('--num_head_c', type=int, default=24)
parser.add_argument('--head_dim', type=int, default=25)
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--step', type=int, default=2)
parser.add_argument('--n_pop', type=int, default=5)
parser.add_argument('--k_pop', type=int, default=5)
parser.add_argument('--k_can', type=int, default=5)
parser.add_argument('--aug', type=bool, default=True)
parser.add_argument('--save_path', default='model_save', help='save model root path')
parser.add_argument('--save_epochs', default=[i for i in range(20)], type=list)
opt = parser.parse_args()
print(opt)

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
if opt.save_path is not None:
    save_path = opt.save_path + '/' + opt.dataset
    save_dir = save_path + '/' + 'aug=' + str(opt.aug) + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print('save dir: ', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def main():
    init_seed(2023)

    if opt.dataset == 'MIND-small':
        n_news = 65239
        n_topic = 19
        n_subtopic = 271
    else:
        n_news = 104152
        n_topic = 19
        n_subtopic = 286

    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    news2topic = pickle.load(open('../datasets/' + opt.dataset + '/news2topic.txt', 'rb'))
    news2subtopic = pickle.load(open('../datasets/' + opt.dataset + '/news2subtopic.txt', 'rb'))
    word_embeddings = pickle.load(open('../datasets/' + opt.dataset + '/word_embeddings.txt', 'rb'))
    news_title_text = pickle.load(open('../datasets/' + opt.dataset + '/news_title_text.txt', 'rb'))
    semantic_similar_news = pickle.load(open('../datasets/' + opt.dataset + '/semantic_similar_news-' + str(opt.k_can) + '.pkl', 'rb'))
    if opt.aug:
        news_clickNum_label = pickle.load(open('../datasets/' + opt.dataset + '/news_clickNum_aug' + str(opt.k_pop) + '_label' + str(opt.n_pop) + '.txt', 'rb'))
    else:
        news_clickNum_label = pickle.load(open('../datasets/' + opt.dataset + '/news_clickNum_label' + str(opt.n_pop) + '.txt', 'rb'))

    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=True)
    model = P_SHE(device, opt, word_embeddings, n_topic, n_subtopic, opt.n_pop)
    model.initialize()
    model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_dc_epoch, gamma=opt.lr_dc)

    start = time.time()
    best_result = [0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------')
        print('epoch:', epoch)
        auc_tes, mrr_tes, ndcg5_tes, ndcg10_tes = \
            train_test(model, train_data, test_data, news_title_text, news2topic, news2subtopic, semantic_similar_news, news_clickNum_label, optimizer)

        if opt.save_path is not None and epoch in opt.save_epochs:
            save_file = save_dir + '/epoch-' + str(epoch) + '.pt'
            torch.save(model, save_file)
            print('save success! :)')

        flag = 0
        if auc_tes >= best_result[0]:
            best_result[0] = auc_tes
            best_epoch[0] = epoch
            flag = 1
        if mrr_tes >= best_result[1]:
            best_result[1] = mrr_tes
            best_epoch[1] = epoch
            flag = 1
        if ndcg5_tes >= best_result[2]:
            best_result[2] = ndcg5_tes
            best_epoch[2] = epoch
            flag = 1
        if ndcg10_tes >= best_result[3]:
            best_result[3] = ndcg10_tes
            best_epoch[3] = epoch
            flag = 1
        print('Best Result:')
        print('\tAuc:\t%.4f\tMRR:\t%.4f\tNDCG@5:\t%.4f\tNDCG@10:\t%.4f\tEpoch:\t%d,\t%d,\t%d,\t%d'
              % (best_result[0], best_result[1], best_result[2], best_result[3], best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
        scheduler.step()

    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


def train_test(model, train_data, test_data, news_title_text, news2topic, news2subtopic, semantic_similar_news, news_clickNum_label, optimizer):
    print('start training:', datetime.datetime.now())
    model.train()
    total_loss = []
    train_slices = train_data.generate_batch(opt.batchSize_train)
    for index in tqdm.tqdm(train_slices):
        optimizer.zero_grad()
        scores, targets, targets_neg_mask = \
            forward(model, index, train_data, news_title_text, news2topic, news2subtopic, semantic_similar_news, news_clickNum_label, device)
        targets = torch.where(targets != 0)[1]
        loss = model.loss_function(scores, targets)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    print('Loss:\t%.6f\tlr:\t%0.8f' % (np.mean(total_loss), optimizer.state_dict()['param_groups'][0]['lr']))

    print('start predicting:', datetime.datetime.now())
    model.eval()
    test_slices = test_data.generate_batch(opt.batchSize_test)
    with torch.no_grad():
        tes_scores = []
        tes_labels = []
        tes_lengths = []
        for index in tqdm.tqdm(test_slices):
            te_scores_items, tes_targets, targets_neg_lengths = \
                forward(model, index, test_data, news_title_text, news2topic, news2subtopic, semantic_similar_news, news_clickNum_label, device)
            tes_scores += te_scores_items.cpu().numpy().tolist()
            tes_labels += tes_targets.cpu().numpy().tolist()
            tes_lengths += targets_neg_lengths.cpu().numpy().tolist()
        predicts = []
        truths = []
        for ss_scores, ll_label, length in zip(tes_scores, tes_labels, tes_lengths):
            ss = ss_scores[:int(length)]
            ll = ll_label[:int(length)]
            sl_zip = sorted(zip(ss, ll), key=lambda x: x[0], reverse=True)
            sort_s, sort_l = zip(*sl_zip)
            predicts.append(list(range(1, len(sort_s) + 1, 1)))
            truths.append(sort_l)
        auc_tes, mrr_tes, ndcg5_tes, ndcg10_tes = evaluate(predicts, truths)
        print('AUC: %0.4f\tMRR: %0.4f\tNDCG5: %0.4f\tNDCG10: %0.4f' %
              (auc_tes * 100, mrr_tes * 100, ndcg5_tes * 100, ndcg10_tes * 100))

        return auc_tes * 100, mrr_tes * 100, ndcg5_tes * 100, ndcg10_tes * 100


def forward(model, index, data, news_title_text, news2topic, news2subtopic, semantic_similar_news, news_clickNum_label, device):
    history_padding, candidate_padding, label_padding = data.get_slice_sess_mask(index)
    neighbour_padding = semantic_similar_news[candidate_padding]
    history_mask = np.where(history_padding == 0, np.zeros_like(history_padding), np.ones_like(history_padding))

    history_topic = news2topic[history_padding]
    candidate_topic = news2topic[candidate_padding]
    neighbour_topic = news2topic[neighbour_padding]
    history_subtopic = news2subtopic[history_padding]
    candidate_subtopic = news2subtopic[candidate_padding]
    neighbour_subtopic = news2subtopic[neighbour_padding]

    candidate_clickNum_label = news_clickNum_label[candidate_padding]

    history_title_word = news_title_text[history_padding]
    candidate_title_word = news_title_text[candidate_padding]
    neighbour_title_word = news_title_text[neighbour_padding]
    history_title_word_mask = np.where(history_title_word == 0, np.zeros_like(history_title_word), np.ones_like(history_title_word))
    candidate_title_word_mask = np.where(candidate_title_word == 0, np.zeros_like(candidate_title_word), np.ones_like(candidate_title_word))
    neighbour_title_word_mask = np.where(neighbour_title_word == 0, np.zeros_like(neighbour_title_word), np.ones_like(neighbour_title_word))

    alias_inputs_topic, alias_inputs_subtopic, topics, subtopics, adj = construct_graph(history_topic, history_subtopic)

    # Tensor
    history_mask = torch.FloatTensor(history_mask).to(device)  # bs * L
    candidate_padding = torch.LongTensor(candidate_padding).to(device)
    label_padding = torch.LongTensor(label_padding).to(device)

    candidate_topic = torch.LongTensor(candidate_topic).to(device)
    neighbour_topic = torch.LongTensor(neighbour_topic).to(device)
    candidate_subtopic = torch.LongTensor(candidate_subtopic).to(device)
    neighbour_subtopic = torch.LongTensor(neighbour_subtopic).to(device)

    candidate_clickNum_label = torch.LongTensor(candidate_clickNum_label).to(device)

    history_title_word = torch.LongTensor(history_title_word).to(device)
    history_title_word_mask = torch.LongTensor(history_title_word_mask).to(device)
    candidate_title_word = torch.LongTensor(candidate_title_word).to(device)
    candidate_title_word_mask = torch.LongTensor(candidate_title_word_mask).to(device)
    neighbour_title_word = torch.LongTensor(neighbour_title_word).to(device)
    neighbour_title_word_mask = torch.LongTensor(neighbour_title_word_mask).to(device)

    alias_inputs_topic = torch.LongTensor(alias_inputs_topic).to(device)
    alias_inputs_subtopic = torch.LongTensor(alias_inputs_subtopic).to(device)
    topics = torch.LongTensor(topics).to(device)
    subtopics = torch.LongTensor(subtopics).to(device)
    adj = torch.LongTensor(adj).to(device)

    scores, candidate_lengths\
        = model(history_mask, candidate_padding,
                history_title_word, history_title_word_mask,
                candidate_title_word, candidate_title_word_mask,
                neighbour_title_word, neighbour_title_word_mask,
                candidate_topic, neighbour_topic,
                candidate_subtopic, neighbour_subtopic,
                candidate_clickNum_label,
                alias_inputs_topic, alias_inputs_subtopic, topics, subtopics, adj)

    return scores, label_padding, candidate_lengths


if __name__ == '__main__':
    main()
