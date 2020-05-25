import torch as t
import torch.nn as nn


class Attention(nn.Module):
    """
    attention采用的key-value对齐方式
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 attention_size,
                 ):
        super(Attention, self).__init__()

        self.linear_q = nn.Sequential(nn.Linear(input_size, attention_size), nn.ReLU())  # query 线性变换
        self.linear_k = nn.Sequential(nn.Linear(hidden_size, attention_size), nn.ReLU())  # key 线性变换
        self.linear_v = nn.Sequential(nn.Linear(hidden_size, attention_size), nn.ReLU())  # value 线性变换

        self.attention_size = attention_size

    def forward(self, inp, hiddens, mask):
        """

        :param inp: 上一步的输出[batch_size, 1, input_size]
        :param hiddens: encoder的输出[batch, seq_len, hidden_size]
        :param mask: [batch_size, seq_len.max+1]
        :return:
        """

        query = self.linear_q(inp)  # [bsz, 1, attn_size]
        key = self.linear_k(hiddens)  # [bsz, seq_len, attn_size]
        value = self.linear_v(hiddens)  # [bsz, seq_len, attn_size]

        len_inp = inp.size(1)

        weight = query.bmm(key.transpose(1, 2))  # [batch_size, 1, seq_len]

        # 当输入向量的维度比较高时，点积模型的值通常有比较大的方差， 从而导致 softmax 函数的梯度会比较小
        # scaled product 可以很好的缓解这个问题
        weight = weight / (self.attention_size ** 0.5)  # scaled
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, len_inp, 1)
            weight = weight.masked_fill(mask == 0, t.tensor(float('-inf')))
        weight = weight.softmax(-1)

        attention = weight.bmm(value)  # [bsz, 1, attn_size]
        return attention, weight
