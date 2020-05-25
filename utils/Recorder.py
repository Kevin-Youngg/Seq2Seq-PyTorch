import time
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data.data_utils import id2sentence
from data.data_utils import UNK_ID, GO_ID, EOS_ID, PAD_ID


class Recorder:
    def __init__(self, args, writer, ids2word):

        self.timestamp = args.timestamp
        self.log_interval = args.log_interval
        self.writer = writer
        self.ids2word = ids2word
        self.batch_size = args.batch_size
        print(f'Record {self.timestamp}')

    def epoch_start(self, epoch_idx, mode, data_size):
        """
        每个epoch开始时记录，epcoh和epoch开始时间
        :param epoch_idx:
        :param is_train:
        :param data_size:
        :return:
        """
        self.epoch_idx = epoch_idx
        self.mode = mode
        # self.dataset_size = len(loader.dataset) if not self.distributed else len(loader.sampler) * self.batch_access
        self.dataset_size = data_size
        self.epoch_loss = 0
        self.epoch_ppl = 0
        self.epoch_bleu = 0
        self.epoch_start_time = time.time()
        self.batch_start_time = time.time()

    def batch_end(self, batch_idx, batch_size, loss, ppl):
        """
        batch结束时记录，
        :param batch_idx:
        :param batch_size:
        :param loss:
        :param ppl:
        :return:
        """

        self.batch_end_time = time.time()
        self.batch_loss = loss
        self.batch_ppl = ppl
        self.epoch_loss += self.batch_loss * batch_size
        self.epoch_ppl += self.batch_ppl * batch_size
        self.batch_time = self.batch_end_time - self.batch_start_time
        self.batch_start_time = time.time()

        if self.mode == 'train' and (batch_idx + 1) % self.log_interval == 0:
            # 输出batch训练的信息
            print('Train: batch {}/epoch {} [{}/{}({:.0f}%)] loss:{:.4f} ppl:{:.4f}/ Time:{:.4f}'.format(
                batch_idx+1,
                self.epoch_idx,
                batch_idx * batch_size,
                self.dataset_size,
                100. * batch_idx * batch_size / self.dataset_size,
                self.batch_loss,
                self.batch_ppl,
                self.batch_time))

            # 把batch训练的信息记录在本地
            batch_record_idx = (self.epoch_idx - 1) * (
                    self.dataset_size // (self.log_interval * self.batch_size)) + batch_idx // self.log_interval
            self.writer.add_scalar(f'{self.mode}-Batch loss', self.batch_loss, batch_record_idx)
            self.writer.add_scalar(f'{self.mode}-Batch perplexity', self.batch_ppl, batch_record_idx)
            self.writer.add_scalar(f'{self.mode}-Batch time', self.batch_time, batch_record_idx)

    def epoch_end(self):
        """
        epoch结束时记录， epoch的loss和ppl
        :return:
        """
        self.epoch_end_time = time.time()
        self.epoch_time = self.epoch_end_time - self.epoch_start_time
        print('====> {}: {} epochs Average loss: {:.4f} ppl:{:.4f}/ Time: {:.4f}'.format(
            self.mode,
            self.epoch_idx,
            self.epoch_loss / self.dataset_size,
            self.epoch_ppl / self.dataset_size,
            self.epoch_time))
        self.writer.add_scalar(f'{self.mode}-Epoch loss', self.epoch_loss / self.dataset_size, self.epoch_idx)
        self.writer.add_scalar(f'{self.mode}-Epoch perplexity', self.epoch_ppl / self.dataset_size, self.epoch_idx)
        self.writer.add_scalar(f'{self.mode}-Epoch bleu', self.epoch_bleu / self.dataset_size, self.epoch_idx)
        self.writer.add_scalar(f'{self.mode}-Epoch time', self.epoch_time, self.epoch_idx)

    def log_text(self, encoder_inputs, decoder_inputs, output_symbols):
        print('in log_text')
        if not self.mode == 'train':
            # if self.mode == 'eval':
            #     n = min(encoder_inputs.size()[0], 8)
            #     encoder_inputs = encoder_inputs[:n]
            #     decoder_inputs = decoder_inputs[:n]
            #     output_symbols = output_symbols[:]
            smooth = SmoothingFunction()
            text_all = []
            for post, ref, response in zip(encoder_inputs, decoder_inputs, output_symbols):
                texts = []
                if PAD_ID in post:
                    post = post[:post.index(PAD_ID)]
                if EOS_ID in response:
                    response = response[:response.index(EOS_ID)]
                if EOS_ID in ref:
                    ref = ref[:ref.index(EOS_ID)]

                self.epoch_bleu += sentence_bleu(references=[ref], hypothesis=response, weights=[1.0, 0.0, 0.0, 0.0],
                                                 smoothing_function=smooth.method1)
                texts.append('post: ' +id2sentence(post, self.ids2word))
                texts.append('response: '+id2sentence(response, self.ids2word))
                texts.append('ref: '+id2sentence(ref, self.ids2word))

                text_all.append('\n'.join(texts))
            if self.mode == 'eval':
                n = min(len(encoder_inputs), 8)
                text_all = text_all[:n]
            text_all = '\n\n'.join(text_all)
            print(text_all)
            self.writer.add_text(f'Pre_{self.mode}', text_all, self.epoch_idx)
