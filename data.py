import numpy as np


class Data(object):
    def __init__(self, data, shuffle):
        self.data = data
        self.sessions = []
        for seq in data[2]:
            tmp = [i for i in seq if i != 0]
            self.sessions.append(tmp)
        self.sessions = np.asarray(self.sessions)
        self.candidates = np.asarray(data[0])
        self.labels = np.asarray(data[1])
        self.length = len(self.sessions)
        self.shuffle = shuffle

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.sessions = self.sessions[shuffled_arg]
            self.candidates = self.candidates[shuffled_arg]
            self.labels = self.labels[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice_sess_mask(self, index):
        sessions = self.sessions[index]
        candidates = self.candidates[index]
        labels = self.labels[index]

        lengths = []
        for sess in sessions:
            lengths.append(len(sess))
        max_length = max(lengths)
        inp_sess_padding = self.zero_padding_mask(sessions, max_length)

        candidate_lengths = []
        for can in candidates:
            candidate_lengths.append(len(can))
        max_neg_length = max(candidate_lengths)
        candidate_padding = self.zero_padding_mask(candidates, max_neg_length)

        lab_lengths = []
        for lab in labels:
            lab_lengths.append(len(lab))
        max_lab_length = max(lab_lengths)
        label_padding = self.zero_padding_mask(labels, max_lab_length)

        return inp_sess_padding, candidate_padding, label_padding

    def zero_padding_mask(self, data, max_length):
        # 返回padding后会话矩阵的mask，mask_inf在会话有物品的地方为0，没有物品的地方为-inf
        out_data = np.zeros((len(data), max_length), dtype=np.int)
        for i in range(len(data)):
            out_data[i, :len(data[i])] = data[i]
        return out_data
