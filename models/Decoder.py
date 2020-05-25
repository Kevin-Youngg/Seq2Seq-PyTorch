import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=2,
                 dropout=0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            batch_first=True
        )

    def forward(self,
                inputs,
                hidden_state):
        '''

        :param inputs:[batch_size, 1, input_size]
        :param hidden_state: [layers, batch_size, hidden_size]
        :return: outputs[batch_size, 1, hidden_size]
                final_state[num_layers, batch_size, hidden_size]
        '''

        outputs, final_state = self.rnn(inputs, hidden_state)
        return outputs, final_state
