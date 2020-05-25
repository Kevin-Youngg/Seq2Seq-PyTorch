import os

# 特殊符号
_PAD = '_PAD'
_GO = '_GO'
_EOS = '_EOS'
_UNK = '_UNK'
_KB = '_KB'
_START_VOCAB = [_PAD, _GO, _EOS, _UNK, _KB]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
KB_ID = 4


def create_vocabulary(vocabulary_path, data_path, min_count, tokenizer=None):
    '''
    生成/data/vocab文件
    :param vocabulary_path: vocab存放目录
    :param data_path: 原始数据路径
    :param min_count: 最小词频
    :param tokenizer: token提取函数
    :return:
    '''
    vocab = {}

    with open(data_path, mode='r', encoding='utf-8') as f:
        for line in f:
            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
            for word in tokens:
                if word in vocab:
                    vocab[word] += 1
                elif word not in vocab:
                    vocab[word] = 1
    # 词表按词频排序
    vocab_list = sorted(vocab, key=vocab.get, reverse=True)
    final_vocab = _START_VOCAB
    # 按预设词频剪裁
    for word in vocab_list:
        if vocab[word] >= min_count:
            final_vocab.append(word)
    with open(vocabulary_path, mode='w', encoding='utf-8') as vocab_file:
        for w in final_vocab:
            vocab_file.write(w + '\n')


def basic_tokenizer(sentence):
    # maybe还可以用subword
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.append(space_separated_fragment)

    return [w for w in words if w]


def initialize_by_vocabulary(vocabulary_path):
    """
    根据词典生成wrod2int, int2word字典
    :param vocabulary_path: vocab路径
    :return:
    """
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path, mode='r', encoding='utf8') as f:
            rev_vocab.extend(f.readlines())
            # 与 append的区别
        rev_vocab = [line.strip() for line in rev_vocab]
        # !!!
        word2int = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        int2word = dict([(x, y) for (x, y) in enumerate(rev_vocab)])

        print('vocab_size:', len(word2int))
        return word2int, int2word, rev_vocab
    else:
        raise ValueError(f'Vocabulary file {vocabulary_path} not found.')


def sentence2id(sentence, word2int, tokenizer=None):
    """
     句子转id
    :param sentence:
    :param word2int:
    :param tokenizer:
    :return:
    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [word2int.get(w, UNK_ID) for w in words]


def id2sentence(id_list, int2word):
    """
    id转句子
    :param id_list:
    :param int2word:
    :return:
    """
    return ' '.join([int2word.get(index, UNK_ID) for index in id_list])


def data2ids(data_path, target_path, vocabulary_path, tokenizer=None):
    """
    将数据处理成ids
    :param data_path: 原始数据路径
    :param target_path: 存放生成ids的文件
    :param vocabulary_path: vocab
    :param tokenizer: token提取函数
    :return:
    """
    if not os.path.exists(target_path):
        word2int, _, _ = initialize_by_vocabulary(vocabulary_path)
        with open(data_path, mode='r', encoding='utf-8') as data_file:
            with open(target_path, mode='w', encoding='utf-8') as tokens_file:
                for line in data_file:
                    token_ids = sentence2id(line, word2int, tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + '\n')
    else:
        print(f'{target_path} has exited.')


def read_data(data_path, min_len=3, max_len=40, max_size=0):
    """
    读取ids数据，构造数据集（dataset中的set）
    :param data_path: 存放已经转换为ids的源数据的目录
    :param min_len: 句子最小长度
    :param max_len: 句子最大长度
    :param max_size:
    :return:
    """
    # 格式化后的数据集,每个数据格式为[source_ids, target_ids]
    dataset = []
    # 保留句子长度范围
    sentence_len_range = range(min_len, max_len)
    with open(data_path, mode='r') as data_file:
        source = data_file.readline()
        target = data_file.readline()
        counter = 0
        while source and target and (not max_size or counter < max_size):  # 没有最大数量限制或符合要求
            counter += 1
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            target_ids.append(EOS_ID)

            if len(source_ids) in sentence_len_range and len(target_ids) in sentence_len_range:
                dataset.append([source_ids, target_ids])
            source = data_file.readline()
            target = data_file.readline()

    print('read_data: {la} source data, {la} target data'.format(la=len(dataset)))
    return dataset


if __name__ == '__main__':
    data_dir = './'

    print(f'{data_dir} hello')
    vocab_path = os.path.join(data_dir, 'vocab')
    create_vocabulary(vocab_path, data_path=os.path.join(data_dir, 'train.txt'), min_count=3)
    train_data_path = os.path.join(data_dir, 'train.txt')
    valid_data_path = os.path.join(data_dir, 'valid.txt')
    test_data_path = os.path.join(data_dir, 'test.txt')
    train_target_path = os.path.join(data_dir, 'train_ids.txt')
    valid_target_path = os.path.join(data_dir, 'valid_ids.txt')
    test_target_path = os.path.join(data_dir, 'test_ids.txt')
    data2ids(train_data_path, train_target_path, vocab_path)
    data2ids(test_data_path, test_target_path, vocab_path)
    data2ids(valid_data_path, valid_target_path, vocab_path)
