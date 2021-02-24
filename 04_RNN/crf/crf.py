import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


def argmax(vec):
    # 返回行最大值得下标
    _, idx = torch.max(vec, 1)
    return idx.item()


# https://blog.csdn.net/zziahgf/article/details/78489562
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# 训练数据
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

# 超参
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# 字典
word_to_id = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)

tag_to_id = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
id_to_tag = {value: key for key, value in tag_to_id.items()}


def prepare_sequence(seq, to_id):
    idxs = [to_id[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_id, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_id = tag_to_id
        self.tagset_size = len(tag_to_id)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 转移矩阵，第i行第j列，表示从状态j转移到状态i的概率
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # 添加约束，不能从开始标记，直接转到结束标记， 也不能从结束标记转移到开始标记
        self.transitions.data[tag_to_id[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_id[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)  # 1 x tagset_size
        init_alphas[0][self.tag_to_id[START_TAG]] = 0.
        forward_var = init_alphas

        # 每个字
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)  # 当前字，是这个tag的概率
                trans_score = self.transitions[next_tag].view(1, -1)  # 从别的状态转到next_tag的概率
                next_tag_var = forward_var + trans_score + emit_score  # 从上一个状态，走一步走到这个状态的概率，乘以这一个字，就是这个状态的概率
                alphas_t.append(log_sum_exp(next_tag_var).view(1)) # 这个字，是这个tag的总分
            forward_var = torch.cat(alphas_t).view(1, -1) # cat 一下，作为下一字的初始状态
        terminal_var = forward_var + self.transitions[self.tag_to_id[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        """ 获取lstm的feature
        """
        self.hidden = self.init_hidden()
        # 获取word embedding
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # lstm
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # 展开
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)  # sentence_length x hidden_size
        # 预测标签的feature
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # 给定标记序列，计算其得分
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_id[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            # 当前分数x状态tag[i] 转到下一个状态 tag[i+1]x是这个状态的概率
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_id[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_id[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # 最大位置状态
            viterbivars_t = []  # 最大位置概率值

            for next_tag in range(self.tagset_size):
                # 往前走一步
                next_tag_var = forward_var + self.transitions[next_tag]
                # 拿到这一步的最大值
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                # 拿到，走到该状态的概率值
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 一步之后的状态
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 再往前走一步，到终点
        terminal_var = forward_var + self.transitions[self.tag_to_id[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 找到最优路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_id[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)  # sentence length x embedding size
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


model = BiLSTM_CRF(len(word_to_id), tag_to_id, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

for epoch in range(1):
    for sentence, tags in training_data:
        optimizer.zero_grad()

        input_sentence = torch.tensor([word_to_id[w] for w in sentence], dtype=torch.long)
        target_label = torch.tensor([tag_to_id[t] for t in tags], dtype=torch.long)
        loss = model.neg_log_likelihood(input_sentence, target_label)

        loss.backward()
        optimizer.step()

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_id)
    print(model(precheck_sent))
    print([tag_to_id[t] for t in training_data[0][1]])
