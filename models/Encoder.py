import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=2,
                 bidirectional=False,
                 dropout=0):
        super().__init__()

        self.bidirectional = bidirectional

        # bi_hidden_size = 2 * uni_hidden_size
        if bidirectional:
            assert hidden_size % 2 == 0
            hidden_size = hidden_size // 2
        else:
            hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=(0 if num_layers == 1 else dropout),
            batch_first=True
        )

    def forward(self,
                inputs,
                length):
        """

        :param self:
        :param inputs:[batch_size, seq_len, input_size]
        :param length:[batch_size]
        :return:outputs: [batch_size, seq_len, hidden_size]
                final_state: [num_layers, batch_size, hidden_size]
        """

        # !!!固定格式，传入网络前，先pack后pad
        inputs = pack_padded_sequence(inputs, length, enforce_sorted=False, batch_first=True)
        outputs, final_state = self.rnn(inputs)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        #output = pad_packed_sequence(output, batch_first=True)[0]

        #如果是双向的， output=[batch_size, seq_len, hidden_size*2]
        #final_state=[2*num_layers, batch_size, hidden_size]
        if self.bidirectional:
            final_state_forward = final_state[0::2, :, :]
            final_state_backward = final_state[1::2, :, :]

            final_state = t.cat([final_state_forward, final_state_backward], dim=2)

        return outputs, final_state

