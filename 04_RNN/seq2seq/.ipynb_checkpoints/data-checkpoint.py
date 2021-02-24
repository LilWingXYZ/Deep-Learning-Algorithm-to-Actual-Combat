import re
import unicodedata
from io import open
import jieba
import torch

from .config import MAX_LENGTH, device, EOS_token


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalize_string(lang, item):
    if lang == "chi":
        item = jieba.cut(item, cut_all=False)
        item = " ".join(item)
    s = unicode_to_ascii(item.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(lang, reverse=False):
    print("Reading lines...")
    lang1, lang2 = lang
    # 读数据的全部行,然后按照\n 去切
    lines = open(f'data/{lang1}-{lang2}.txt', encoding='utf-8').read().strip().split('\n')

    # 对于每一行, 先根据\t 切成a, b 两种语言的 string, 然后对这两种语言,都进行 normalize
    # 最终得到的是一个二维的 list, 第一维是行的 list,第二位维是两种语言的 list
    pairs = [[normalize_string(lang[index], item) for index, item in enumerate(l.split('\t'))] for l in lines]

    # 控制谁是目标语言,谁是源语言
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
    )


def filter_pair(p):
    """
    过滤的条件是,源语言和目标语言的句子长度都小于 MAX_LENGTH, 然后,目标语言要以 eng_prefixes 开头
    :param p:
    :return:
    """
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes)


def filter_pairs(pairs):
    """
    删选 paris 的一对对的数据,将符合条件的拿出来
    :param pairs:
    :return:
    """
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs((lang1, lang2), reverse)
    print("Read %s sentence lpairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexes_from_sentence(lang, sentence):
    """
    将句子中的 word token 化
    :param lang:
    :param sentence:
    :return:
    """
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(input_lang, output_lang, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, target_tensor
