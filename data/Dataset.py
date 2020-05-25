from .data_utils import *
from torch.utils.data import DataLoader, Dataset
import torch as t
from torch.nn.utils.rnn import pad_sequence


class Set(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Loader:
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=False,
                 use_gpu=False,
                 num_workers=0):
        self.loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.process
        )
        self.PAD_ID = PAD_ID
        self.GO_ID = GO_ID
        self.batch_size = batch_size,
        self.device = t.device('cuda') if use_gpu else t.device('cpu')

    def process(self, batch):
        """
        每次加载batch执行的数据处理
        :param batch:
        :return:
        """
        encoder_inputs = []
        seq_len = []
        decoder_inputs = []
        # batch中的元素[source_ids, target_ids]
        for i, batch_item in enumerate(batch):
            source_ids, target_ids = batch_item
            # 把一条post的id转成tensor，保存到列表，方便调用pad_sequence进行填充操作
            encoder_inputs.append(t.tensor(source_ids))
            seq_len.append(len(source_ids))
            # response加上开始符
            decoder_inputs.append(t.tensor([self.GO_ID] + target_ids))

        # padding方便一个batch作为整体训练
        encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=self.PAD_ID)
        seq_len = t.tensor(seq_len)
        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=self.PAD_ID)


        # 用来统计句子长度,之后计算损失时用到
        weights = t.ones(size=decoder_inputs[:, 1:].size())  # [batch_size, max_len-1]
        weights.masked_fill_(decoder_inputs[:, 1:].eq(self.PAD_ID), value=0)

        #这里注意哦，如果返回转移到cuda上的数据，如果num_wokers>0的话会报错，
        # 因为pytorch多线程使用spawn，python多线程使用fork，
        # 所以建议在将数据送入网络前再转移到cuda上
        # return encoder_inputs.to(self.device), seq_len.to(self.device), decoder_inputs.to(self.device), weights.to(self.device)
        return encoder_inputs, seq_len, decoder_inputs, weights
