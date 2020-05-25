import torch as t
import torch.nn as nn
from .Encoder import Encoder
from .Decoder import Decoder
from .Attention import Attention
import torch.nn.functional as F
import random
from queue import Queue
from data.data_utils import EOS_ID, GO_ID
import numpy as np


class Seq2Seq(nn.Module):
    def __init__(self,
                 embed_size,  # 词向量大小
                 enc_dec_output_size,  # hiiden_size
                 attn_size,
                 num_layers,
                 bidirectional,
                 vocab_size,
                 dropout=0,
                 use_gpu=False):

        super(Seq2Seq, self).__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # encoder
        self.encoder = Encoder(
            input_size=embed_size,
            hidden_size=enc_dec_output_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )
        # decoder
        self.decoder = Decoder(
            input_size=embed_size,
            hidden_size=enc_dec_output_size,
            num_layers=num_layers,
            dropout=dropout
        )
        # attention
        self.attention = Attention(
            input_size=enc_dec_output_size,
            hidden_size=enc_dec_output_size,
            attention_size=attn_size,
        )
        self.dropout = nn.Dropout(dropout)
        # concat
        self.concat = nn.Linear(enc_dec_output_size + attn_size, enc_dec_output_size)
        # projection
        self.out_project = nn.Linear(enc_dec_output_size, vocab_size)
        self.attn_size = attn_size
        self.vocab_size = vocab_size
        self.device = t.device('cuda') if use_gpu else t.device('cpu')

    def forward(self,
                encoder_inputs,
                seq_len,
                decoder_inputs,
                mode,
                max_len=0,
                beam_search=False,
                beam_size=1,
                topk=1,
                teacher_forcing_ratio=0):
        # 获得输入的词向量表示
        emb_enc_inputs = self.embedding(encoder_inputs)

        # 送入encoder
        enc_outputs, enc_final_hidden = self.encoder(emb_enc_inputs, seq_len)
        batch_size = seq_len.size(0)
        # !!!这个地方转换的太巧妙了！！
        # seq_len为原始数据中post的长度
        # +1的原因是下标从0开始
        # seq_len [bsz]
        # F.one_hot(seq_len, seq_len.max() + 1) ==> [bsz, seq_len.max+1]
        # for example seq_len = [1, 2, 3]
        # F.one_hot(seq_len, seq_len.max() + 1)
        # [0, 1, 0, 0]
        # [0, 0, 1, 0]
        # [0, 0, 0, 1]
        # F.one_hot(seq_len, seq_len.max() + 1).cumsum(1)
        # [0, 1, 1, 1]
        # [0, 0, 1, 1]
        # [0, 0, 0, 1]
        # 1 - F.one_hot(seq_len, seq_len.max() + 1).cumsum(1)
        # [1, 0, 0, 0]
        # [1, 1, 0, 0]
        # [1, 1, 1, 0]
        # !!!!!!! +1的原因是要包括0 [batch_size] -> [batch_size, seq_len.max+1] ->[batch_size,seq_len.max]
        attn_mask = (1 - F.one_hot(seq_len, seq_len.max() + 1).cumsum(1))[:, :-1].bool()

        dec_outputs = []
        output_symbols = []
        # 将编码器最后一步的隐状态作为decoder初始隐状态
        hidden = enc_final_hidden
        # 初始化attention向量

        attn = t.zeros(batch_size, 1, self.attn_size).to(self.device)

        embed_dec_inputs = self.embedding(decoder_inputs)
        emb_inp = embed_dec_inputs[:, 0:1]
        # train or val
        if mode == 'train':
            for i in range(embed_dec_inputs.size(1)):
                dec_input = emb_inp
                dec_output, hidden = self.decoder(dec_input, hidden)
                # dec_output [batch_size, 1, hidden_size]
                attn, _ = self.attention(dec_output, enc_outputs, attn_mask)
                # attn [bsz, 1, attn_size[

                concat_input = t.cat([dec_output, attn], dim=2)

                concat_output = self.concat(concat_input)
                logit = self.out_project(t.tanh(concat_output)).log_softmax(-1)  # [bsz, seq_len, vocab]
                # print('train logit size', logit.size())
                dec_outputs.append(logit)

                output_symbol = logit.argmax(dim=-1)
                output_symbols.append(output_symbol)
                # random的范围为[0,1) 注意这里的问题 是<= or < 一开始这里搞错啦 写的 >=
                teacher_force = random.random() < teacher_forcing_ratio
                emb_inp = embed_dec_inputs[:, i + 1:i + 2] if teacher_force else self.embedding(
                    output_symbol)

            dec_outputs = t.cat(dec_outputs, dim=1)
            output_symbols = t.cat(output_symbols, dim=1)
            # return dec_outputs, output_symbols

        elif mode == 'eval' or mode == 'test':
            if beam_search:
                dec_outputs, output_symbols = self.beamSearch(hidden, enc_outputs, beam_size, topk=topk,
                                                              max_len=max_len,
                                                              attn_mask=attn_mask)
            else:
                dec_outputs, output_symbols = self.greadySearch(embed_dec_inputs[:, 0:1], hidden, enc_outputs,
                                                                max_len=max_len,
                                                                attn_mask=attn_mask)
                # print(output_symbols)

        return dec_outputs, output_symbols

    def greadySearch(self, embed_inp, hidden, enc_outputs, max_len=0, attn_mask=None):
        # print('in greadySearch...')
        dec_outputs = []
        output_symbols = []
        # 搜索到最大长度
        for i in range(max_len):
            dec_input = embed_inp
            dec_output, hidden = self.decoder(dec_input, hidden)

            attn, _ = self.attention(dec_output, enc_outputs, attn_mask)

            concat_input = t.cat([dec_output, attn], dim=2)

            concat_output = self.concat(concat_input)
            logit = self.out_project(t.tanh(concat_output)).log_softmax(-1)
            dec_outputs.append(logit)

            output_symbol = logit.argmax(dim=-1)
            output_symbols.append(output_symbol)

            embed_inp = self.embedding(output_symbol)
        dec_outputs = t.cat(dec_outputs, dim=1)  # [bsz, seq_len, vocab_size]
        output_symbols = t.cat(output_symbols, dim=1)  # [bsz,seq_len[]]
        # print('dec size and out size',dec_outputs.size(), output_symbols.size())
        return dec_outputs, output_symbols

    def beamSearch(self, hiddens, enc_outputs, beam_size=1, topk=1, max_len=0, attn_mask=None):
        # print('in beamSearch...')
        dec_outputs = []
        output_symbols = []
        for index in range(hiddens.size(1)):
            decoder_hidden = hiddens[:, index, :].unsqueeze(
                1).contiguous()  # [2, bsz, hidden_size] ==> [2, 1, hidden_size]
            encoder_output = enc_outputs[index, :, :].unsqueeze(
                0).contiguous()  # [bsz, seq_len, hidden_size] ==>[1, seq_len. hidden_size]
            decoder_attn_mask = attn_mask[index, :].unsqueeze(0).contiguous()  # [bsz. seq_len] ==> [1, seq_len]

            endnodes = []  # 最终节点的位置

            # 开始节点（跟节点） -  hidden vector, logit, previous node, word id, prob, length
            root = BeamSearchNode(decoder_hidden, None, None, GO_ID, 0, 1)
            q = Queue()

            q.put(root)
            # 从跟节点开始层次遍历
            # 每层选出前topk个
            while not q.empty():
                candidates = []  # 每一层可能被拓展的节点
                for i in range(q.qsize()):
                    # 获取节点
                    node = q.get()
                    # 将int转换文tensor，并扩展为[1,1],因为之后要经过embedding层， embedding层的输入为[bsz, seq_len]
                    worid = t.tensor([node.wordid]).unsqueeze(dim=0).to(self.device)
                    # 得到embedding向量
                    decoder_input = self.embedding(worid)  # decoder_input [1, 1, embed_size]
                    decoder_hidden = node.h
                    # 搜索终止条件，到达EOS_I（或许可以只限制max_len，这样后边回溯的输出处理时或许可以不考虑padding了，debug在此处耗费了较多时间)
                    if node.wordid == EOS_ID or node.length >= max_len:
                        endnodes.append((node.eval(), node))
                        # print('search one the len is', node.length)
                        # 如果已经搜索到足够的句子，则退出
                        if len(endnodes) >= beam_size:
                            q.queue.clear()
                            break
                        continue
                    # print('dec_inp size', decoder_input.size())
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                    attn, _ = self.attention(decoder_output, encoder_output, decoder_attn_mask)

                    concat_input = t.cat([decoder_output, attn], dim=2)
                    concat_output = self.concat(concat_input)

                    logit = self.out_project(t.tanh(concat_output)).log_softmax(-1)  # logit[1,1,vocab_size]

                    # 得到topk symbol
                    # log_prob [1, 1, topk], indices [1, 1, topk]
                    log_prob, indices = logit.topk(beam_size, dim=-1)


                    for k in range(beam_size):

                        index = indices[0][0][k].item()
                        log_p = logit[0][0][k].item()

                        # node - hidden vector, logit, previous node, wordid, prob, length
                        child = BeamSearchNode(decoder_hidden, logit, node, index, node.logp + log_p, node.length + 1)
                        score = child.eval()
                        candidates.append((score, child))
                # 候选节点排序
                candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

                #这步不能省略，因为可能遇到在该层遇到了一个EOS，导致candidate数目小于beam_size
                n = min(len(candidates), beam_size)
                for i in range(n):
                    q.put(candidates[i][1])

            # 回溯
            utterances = [] #保存该样本的预测的topk个可能句子的symbol
            logits = []
            endnodes = sorted(endnodes, key=lambda x: x[0], reverse=True)

            for k in range(topk):
                _, n = endnodes[k]

                utterance = []
                logit = []
                utterance.append(n.wordid)
                logit.append(n.logit)
                # print(n.prevNode.wordid)
                while n.prevNode.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)
                    logit.append(n.logit)  # n.logit[1,1,vocab_size] type=tensor
                # 反转
                utterance = utterance[::-1]  # [seq_len]
                logit = logit[::-1]

                # utterance = t.tensor(utterance).unsqueeze(dim=0)
                pad = [0 for _ in range(max_len)]
                pad[:len(utterance)] = utterance
                utterance = pad
                utterances.append(utterance)  # [topk, seq_len]

                # 注意tensor不能有嵌套关系，所以现转换为list最后统一转换为tensor
                # pad为max_len 不要fill为0哦，因为这是预测的回复log_softmax
                logit = t.cat(logit, dim=1)  # logit [1, seq_len, vocab_size]
                #不能填充为-inf，最后计算loss会警告NaN
                p.zeros((logit.size(0), max_len, logit.size(2)), device=logit.device).fill_(float('-inf'))
                pad[:, :logit.size(1), :] = logit
                logit = pad.cpu().numpy().tolist()  # logit [1, max_len, vocab_size]

                logits.append(logit)  # [topk, 1, max_len, vocab_size]

            # 取完topk,保存每个batch中每个样本的的预测结果
            # print('len utterances', np.array(utterances).shape)
            output_symbols.append(t.tensor(utterances).unsqueeze(dim=1))  # [topk,1, seq_len]*bsz
            logits = t.tensor(logits).to(self.device)
            dec_outputs.append(logits)  # [topk, 1, seq_len, vocab_size]*bsz
        # print('123', dec_outputs)
        # print('bsz:', len(dec_outputs), len(output_symbols))
        # print('numpy shape:', dec_outputs[0].size(), output_symbols[0].size())
        output_symbols = t.cat(output_symbols, dim=1)  # [topk, bsz, seq_len]
        dec_outputs = t.cat(dec_outputs, dim=1)  # [topk, bsz, seq_len, vocab_size]
        if topk == 1:
            output_symbols = output_symbols.squeeze(dim=0)
            dec_outputs = dec_outputs.squeeze(dim=0)
        # print('exit bs', dec_outputs.size(), output_symbols.size())
        return dec_outputs, output_symbols

    # 保存模型的结果
    def save(self, save_path, **kwargs):
        # 保存网络参数
        state_dict = {
            'embedding': self.embedding.state_dict(),
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'attention': self.attention.state_dict(),
            'concat': self.concat.state_dict(),
            'out_project': self.out_project.state_dict(),
        }
        # 保存其他自定义的参数，如epoch
        if kwargs:
            for k in kwargs:
                state_dict[k] = kwargs[k]
        t.save(state_dict, save_path)

    def load(self, load_path, return_list=None):
        state_dict = t.load(load_path)
        self.embedding.load_state_dict(state_dict['embedding'])
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder.load_state_dict(state_dict['decoder'])
        self.attention.load_state_dict(state_dict['attention'])
        self.concat.load_state_dict(state_dict['concat'])
        self.out_project.load_state_dict(state_dict['out_project'])

        if return_list:
            return [state_dict[x] for x in return_list]


class BeamSearchNode(object):
    def __init__(self, hiddenstate, logit, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        # self.di = dec_input
        self.h = hiddenstate
        self.logit = logit
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        # 概率缩放
        return self.logp / float(self.length + 1e-6) + alpha * reward
