import numpy as np
import json
import nltk
import pickle
import random
from nltk.tokenize import word_tokenize

np.random.seed(2023)
random.seed(2023)

dataset_name = 'MIND/MINDsmall'
output_dataset = 'MIND-small'
with open(dataset_name + '_train/behaviors.tsv', encoding='utf-8') as f:
    train_behaviors = f.readlines()
with open(dataset_name + '_dev/behaviors.tsv', encoding='utf-8') as f:
    dev_behaviors = f.readlines()
with open(dataset_name + '_train/news.tsv', encoding='utf-8') as f:
    train_newsMetadata = f.readlines()
with open(dataset_name + '_dev/news.tsv', encoding='utf-8') as f:
    dev_newsMetadata = f.readlines()
    newsMetadata = train_newsMetadata + dev_newsMetadata

#
news = {}
for i in newsMetadata:
    line = i.strip('\n').split('\t')
    if line[0] not in news:
        news[line[0]] = [line[1], line[2], line[3]]
#

news = {}
category = {'NULL': 0}
subcategory = {'NULL': 0}
for i in newsMetadata:
    line = i.strip('\n').split('\t')
    if line[0] not in news:
        news[line[0]] = [line[1], line[2], word_tokenize(line[3].lower())]
    if line[1] not in category:
        category[line[1]] = len(category)
    if line[2] not in subcategory:
        subcategory[line[2]] = len(subcategory)
print('Number of news:', len(news))
print('Number of category:', len(category)-1)
print('Number of subcategory:', len(subcategory)-1)

news_index = {'NULL': 0}
for i in news:
    news_index[i] = len(news_index)

    if len(news_index) == 29788 or len(news_index) == 62326 or len(news_index) == 63873 or len(news_index) == 32173 or len(news_index) == 37571 or len(news_index) == 34901 or len(news_index) == 43150 or len(news_index) == 37784 or len(news_index) == 64728:  # 29788 or 62326 or 63873 or 32173 or 37571 or 34901 or 43150 or 37784 or 64728
        print('news_index:', len(news_index))
        print(i)

# word_dict: 'word': [index, feq]
word_dict = {'PADDING': [0, 999999]}
for i in news:
    for j in news[i][2]:
        if j in word_dict:
            word_dict[j][1] += 1
        else:
            word_dict[j] = [len(word_dict), 1]
print('Number of word:', len(word_dict))

word_vector = {}
with open('glove.840B.300d.txt', 'rb') as f:
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        line = line.split()
        word = line[0].decode()
        if len(word) != 0:
            vec = [float(x) for x in line[1:]]
            if word in word_dict:
                word_vector[word] = vec

word_embeddings = [0] * len(word_dict)
in_dict_emb = []
for i in word_vector.keys():
    word_embeddings[word_dict[i][0]] = np.array(word_vector[i], dtype='float32')
    in_dict_emb.append(word_embeddings[word_dict[i][0]])
in_dict_emb = np.array(in_dict_emb, dtype='float32')
mu = np.mean(in_dict_emb, axis=0)
Sigma = np.cov(in_dict_emb.T)
norm = np.random.multivariate_normal(mu, Sigma, 1)  # 给定 均值， 协方差 ,维度1 ，norm.shape= （1，300）

# emb_mat是第一行为padding的总词表
for i in range(len(word_embeddings)):
    if type(word_embeddings[i]) == int:  # word_embeddings[i]在GloVe中找不到词向量
        word_embeddings[i] = np.reshape(norm, 300)
word_embeddings[0] = np.zeros(300, dtype='float32')  # word padding
word_embeddings = np.array(word_embeddings, dtype='float32')
print('Shape of word embeddings:', word_embeddings.shape)

# title到word_id的对应,第一行为0，且每行都padding
news_title = [[0] * 30]
for i in news:
    title = []
    for word in news[i][2]:
        if word in word_dict:
            title.append(word_dict[word][0])
        else:
            print('error!')
    title = title[:30]
    news_title.append(title + [0] * (30 - len(title)))
news_title = np.array(news_title, dtype='int32')

# news到topic的对应,第一个为0
news_topic = [0]
for i in news:
    news_topic.append(category[news[i][0]])
news_topic = np.array(news_topic, dtype='int32')

news_subtopic = [0]
for i in news:
    news_subtopic.append(subcategory[news[i][1]])
news_subtopic = np.array(news_subtopic, dtype='int32')


def negative_sample(negative_news, sample_num):
    # sample_result = []
    # num_news = len(negative_news)
    # if num_news <= sample_num:
    #     for i in range(sample_num):
    #         sample_result.append(negative_news[i % num_news])
    # else:
    #     used_negative_news = set()
    #     for i in range(sample_num):
    #         while True:
    #             k = np.random.randint(0, num_news)
    #             if k not in used_negative_news:
    #                 sample_result.append(negative_news[k])
    #                 used_negative_news.add(k)
    #                 break
    # return sample_result
    if sample_num > len(negative_news):
        return random.sample(negative_news * (sample_num // len(negative_news) + 1), sample_num)
    else:
        return random.sample(negative_news, sample_num)


negative_sample_num = 4
train_valid_split = 0.95

train = train_behaviors[:int(train_valid_split * len(train_behaviors))]
valid = train_behaviors[int(train_valid_split * len(train_behaviors)):]
test = dev_behaviors


train_history = []
train_candidate = []
train_candidate_label = []
for line in train:
    line = line.replace('\n', '').split('\t')
    history = [news_index[x] for x in line[3].split()][-50:]
    pos_news = [news_index[x.split('-')[0]] for x in line[4].split() if x.split('-')[1] == '1']
    neg_news = [news_index[x.split('-')[0]] for x in line[4].split() if x.split('-')[1] == '0']
    if history == [] or pos_news == [] or neg_news == []:
        continue
    for p_news in pos_news:
        candidate = negative_sample(neg_news, negative_sample_num)
        candidate.append(p_news)
        candidate_label = [0] * negative_sample_num + [1]
        candidate_order = list(range(negative_sample_num + 1))
        random.shuffle(candidate_order)
        candidate_shuffle = []
        candidate_label_shuffle = []
        for i in candidate_order:
            candidate_shuffle.append(candidate[i])
            candidate_label_shuffle.append(candidate_label[i])
        train_history.append(history + [0] * (50 - len(history)))
        train_candidate.append(candidate_shuffle)
        train_candidate_label.append(candidate_label_shuffle)
print('train:', len(train_history))


valid_history = []
valid_candidate = []
valid_candidate_label = []
for line in valid:
    line = line.replace('\n', '').split('\t')
    history = [news_index[x] for x in line[3].split()][-50:]
    candidate = [news_index[x.split('-')[0]] for x in line[4].split()]
    candidate_label = [int(x.split('-')[1]) for x in line[4].split()]
    if history == [] or candidate == []:
        continue
    valid_history.append(history + [0] * (50 - len(history)))
    valid_candidate.append(candidate)
    valid_candidate_label.append(candidate_label)
print('valid:', len(valid_history))


test_history = []
test_candidate = []
test_candidate_label = []
test_index = []
for line in test:
    line = line.replace('\n', '').split('\t')
    history = [news_index[x] for x in line[3].split()][-50:]
    candidate = [news_index[x.split('-')[0]] for x in line[4].split()]
    candidate_label = [int(x.split('-')[1]) for x in line[4].split()]
    if history == [] or candidate == []:
        continue
    test_history.append(history + [0] * (50 - len(history)))
    test_candidate.append(candidate)
    test_candidate_label.append(candidate_label)
print('test:', len(test_history))


train = (train_candidate, train_candidate_label, train_history)
valid = (valid_candidate, valid_candidate_label, valid_history)
test = (test_candidate, test_candidate_label, test_history)


pickle.dump(train, open(output_dataset+'/train.txt', 'wb'))
pickle.dump(valid, open(output_dataset+'/valid.txt', 'wb'))
pickle.dump(test, open(output_dataset+'/test.txt', 'wb'))
pickle.dump(news_index, open(output_dataset+'/news_index.txt', 'wb'))
pickle.dump(word_embeddings, open(output_dataset+'/word_embeddings.txt', 'wb'))
pickle.dump(news_title, open(output_dataset+'/news_title_text.txt', 'wb'))
pickle.dump(news_topic, open(output_dataset+'/news2topic.txt', 'wb'))
pickle.dump(news_subtopic, open(output_dataset+'/news2subtopic.txt', 'wb'))
pickle.dump(category, open(output_dataset+'/category.txt', 'wb'))
pickle.dump(subcategory, open(output_dataset+'/subcategory.txt', 'wb'))

# 去除0交互
# MIND-small
# Number of news: 65238
# Number of category: 18
# Number of subcategory: 270
# Number of word: 42055
# Shape of word embeddings: (42055, 300)
# train: 219988
# valid: 7674
# test: 70938
# large
# Number of news: 104151
# Number of category: 18
# Number of subcategory: 285
# Number of word: 52113
# Shape of word embeddings: (52113, 300)
# train: 3147381
# valid: 109382
# test: 365201

# 不去除0交互
# Number of news: 65238
# Number of category: 18
# Number of subcategory: 270
# Number of word: 42055
# Shape of word embeddings: (42055, 300)
# train: 224536
# valid: 7849
# test: 73152
