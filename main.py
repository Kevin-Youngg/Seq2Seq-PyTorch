import os
import datetime
import torch as t
import torch.backends.cudnn as cudnn
from data.data_utils import read_data, initialize_by_vocabulary, PAD_ID, EOS_ID, id2sentence, GO_ID
from data.Dataset import Loader, Set
from models.Seq2Seq import Seq2Seq
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils.Recorder import Recorder
import argparse
import random
from tensorboardX import SummaryWriter

model_dir = 'checkpoints'
data_dir = 'data'
result_dir = 'results'


def train(args, vocab_size):
    # opt._parse(kwarg)
    print('enter train func')
    device = t.device('cuda') if args.use_gpu else t.device('cpu')
    model = Seq2Seq(embed_size=args.embed_size,
                    enc_dec_output_size=args.enc_dec_output_size,
                    attn_size=args.attn_size,
                    num_layers=args.num_layers,
                    bidirectional=args.bidirectional,
                    use_gpu=args.use_gpu,
                    vocab_size=vocab_size).to(device)

    print('Model structure')
    print(model)
    print('The model has %d parameters' % count_parameters(model))

    if args.load_model_path is not None:
        rev_path = os.path.join(model_dir, args.load_model_path)
        if os.path.exists(rev_path):
            print('read in model from', rev_path)
            last_epoch = model.load(load_path=rev_path, return_list=['epoch'])[0]
            start_epoch = last_epoch + 1

    else:
        start_epoch = 1
        last_epoch = -1
    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.scheduler_type == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.exponential_lr_decay, last_epoch=last_epoch)
    elif args.scheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_lr_decay)
    print('read in data')

    # 读取数据
    batch_size = args.batch_size
    train_set = Set(read_data(args.train_data_root))
    valid_set = Set(read_data(args.valid_data_root))
    # 构造dataloader
    train_loader = Loader(train_set,
                          batch_size,
                          shuffle=True,
                          use_gpu=args.use_gpu,
                          num_workers=args.num_workers).loader
    valid_loader = Loader(valid_set,
                          batch_size,
                          shuffle=False,
                          use_gpu=args.use_gpu,
                          num_workers=args.num_workers).loader

    # 统计数据量
    print('data scale:')
    print('train data:', len(train_set), "batch_nums:", len(train_loader))
    print('valid data:', len(valid_set), "batch_nums:", len(valid_loader))

    # train
    print('start training...')
    epochs = args.max_epoch

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        # epoch开始前记录
        recorder.epoch_start(epoch, 'train', len(train_set))

        if args.scheduler_type is not None:
            print(epoch, 'lr={:.10f}'.format(scheduler.get_lr()[0]))
        for batch_id, batch in enumerate(train_loader):
            encoder_inputs, seq_len, decoder_inputs, weights = batch
            encoder_inputs = encoder_inputs.to(device)
            seq_len = seq_len.to(device)
            decoder_inputs = decoder_inputs.to(device)
            weights = weights.to(device)
            encoder_inputs.to(device)
            optimizer.zero_grad()

            # 第三个参数， 最长的句子最后一个token为EOS_I，不需要作为输入，这样可以减少一些计算
            logits, output_symbols = model(encoder_inputs, seq_len, decoder_inputs[:, :-1], mode='train', max_len=None,
                                           teacher_forcing_ratio=args.teacher_forcing_ratio)

            # print('train out',output_symbols)
            # 计算损失
            nll_loss = compute_loss(logits, decoder_inputs[:, 1:], weights)
            # 计算困惑度
            ppl = perplexity(nll_loss)
            # print(nll_loss.item(), ppl.item())
            # 反向传播，更新参数
            nll_loss.backward()
            # 减轻梯度爆炸 小trick
            nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
            optimizer.step()

            recorder.batch_end(batch_id, batch_size, nll_loss, ppl)

        if args.scheduler_type is not None:
            scheduler.step()
        recorder.epoch_end()
        # 保存模型
        if epoch % 5 == 0:
            model.save(os.path.join(model_dir,
                                    f'{args.project}_{datetime.datetime.now().strftime("%y_%m_%d_%H:%M:%S")}_{nll_loss.item()}_{ppl.item()}'),
                       epoch=epoch)
        # 训练一轮后，在验证集上计算loss, ppl
        model.eval()
        with t.no_grad():
            recorder.epoch_start(epoch, 'eval', len(valid_set))
            for batch_id, batch in enumerate(valid_loader):
                encoder_inputs, seq_len, decoder_inputs, weights = batch
                encoder_inputs = encoder_inputs.to(device)
                seq_len = seq_len.to(device)
                decoder_inputs = decoder_inputs.to(device)
                weights = weights.to(device)
                encoder_inputs.to(device)
                logits, output_symbols = model(encoder_inputs, seq_len, decoder_inputs[:, :-1], mode='eval',
                                               max_len=args.max_len,
                                               beam_search=False if args.beam_size == 1 else True,
                                               beam_size=args.beam_size,
                                               topk=args.topk
                                               )
                # print('eval out: ', output_symbols)
                nll_loss = compute_loss(logits, decoder_inputs[:, 1:], weights)
                ppl = perplexity(nll_loss)
                recorder.batch_end(batch_id, batch_size, nll_loss, ppl)
                recorder.log_text(encoder_inputs.tolist(), decoder_inputs[:, 1:].tolist(), output_symbols.tolist())
            recorder.epoch_end()


def test(args, vocab_size):
    device = t.device('cuda') if args.use_gpu else t.device('cpu')
    beam_size = args.beam_size
    topk = args.topk
    rev_model = args.load_model_path
    # print(rev_model)
    model = Seq2Seq(embed_size=args.embed_size,
                    enc_dec_output_size=args.enc_dec_output_size,
                    attn_size=args.attn_size,
                    num_layers=args.num_layers,
                    bidirectional=args.bidirectional,
                    use_gpu=args.use_gpu,
                    vocab_size=vocab_size).to(device)

    assert rev_model is not None

    # 读取已经保存的模型
    rev_path = os.path.join(model_dir, rev_model)
    if os.path.exists(rev_path):
        print('read in model from', rev_path)
        model.load(load_path=rev_path)

    batch_size = args.batch_size
    test_set = Set(read_data(args.test_data_root))
    test_loader = Loader(test_set, batch_size, shuffle=False, use_gpu=args.use_gpu,
                         num_workers=args.num_workers).loader

    model.eval()
    with t.no_grad():
        recorder.epoch_start(0, 'test', len(test_set))
        for batch_id, batch in enumerate(test_loader):
            encoder_inputs, seq_len, decoder_inputs, weights = batch
            encoder_inputs = encoder_inputs.to(device)
            seq_len = seq_len.to(device)
            decoder_inputs = decoder_inputs.to(device)
            weights = weights.to(device)
            encoder_inputs.to(device)
            logits, output_symbols = model(encoder_inputs, seq_len, decoder_inputs[:, :-1], mode='test',
                                           max_len=args.max_len,
                                           beam_search=False if args.beam_size == 1 else True,
                                           beam_size=args.beam_size,
                                           topk=args.topk
                                           )

            nll_loss = compute_loss(logits, decoder_inputs[:, 1:], weights)
            ppl = perplexity(nll_loss)
            recorder.batch_end(batch_id, batch_size, nll_loss, ppl)
            recorder.log_text(encoder_inputs.tolist(), decoder_inputs[:, 1:].tolist(), output_symbols.tolist())
        recorder.epoch_end()


nllloss = nn.NLLLoss(reduction='none', ignore_index=PAD_ID)


def compute_loss(logits, targets, weights):
    """

    :param logits: [bsz,max_len,vocab_size]
    :param targets: [bsz,seq_len]
    :param weights:（可以不用）
    :return:
    """
    max_len = logits.size(1)  # train和 eval、test时的值会不同，train时max_len = real_len, test、eval时可能大于/小于real_len
    batch_size = targets.size(0)
    real_len = targets.size(1)  # 真实回复的长度
    # print('in com', max_len, batch_size, real_len, logits.size())
    if max_len > real_len:  # 如果大于，则剪裁掉多余的部分
        logits = logits[:, :real_len, :]
    elif max_len < real_len:  # 如果小于，则padding
        pad = t.zeros((batch_size, real_len, logits.size(1)), device=logits.device)  # 先定义一个用作padding的零tensor
        pad[:, :real_len, :] = logits  # 将模型输出填充到tensor中
        logits = pad  # 再将padding赋值给logits，作为计算loss的logits

    logits = logits.reshape(batch_size * real_len, -1)
    targets = targets.reshape(-1)
    # 好好理解这里形状的变化 nllloss input [n,c] target[n]
    # 是因为 nll_loss the same shape as target
    # print('int compute_loss, logits.size()', logits.size(), 'targets.size', targets.size())

    nll_loss = F.nll_loss(input=logits, target=targets, reduction='mean', ignore_index=PAD_ID)

    return nll_loss


def perplexity(nll_loss):
    return t.exp(nll_loss)


def count_parameters(model):
    print('count of parameters', sum(p.numel() for p in model.parameters()))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # t.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--train_data_root', type=str, default='./data/test_ids.txt')
    parser.add_argument('--test_data_root', type=str, default='./data/test_ids.txt')
    parser.add_argument('--valid_data_root', type=str, default='./data/valid_ids.txt')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--save_model_dir', type=str, default='./checkpoints')
    parser.add_argument('--project', type=str, default='seq2seq')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%y_%m_%d_%H:%M:%S"))

    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--enc_dec_output_size', type=int, default=256)
    parser.add_argument('--attn_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=40)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler_type', type=str, default=None)
    parser.add_argument('--exponential_lr_decay', type=float, default=0.95)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--step_lr_decay', type=float, default=0.1)
    parser.add_argument('--teacher_forcing_ratio', type=int, default=1)
    parser.add_argument('--max_gradient_norm', type=float, default=5.0)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    print(args.use_gpu, args.bidirectional)
    # 设定随机树种子，确保得到相同的结果
    if args.seed is not None:
        random.seed(args.seed)
        t.manual_seed(args.seed)
        cudnn.deterministic = True

    vocab_path = os.path.join(data_dir, "vocab")
    # 根据常见的词典，得到映射表
    word2id, id2word, vocab = initialize_by_vocabulary(vocab_path)

    # 如果不存在目标路径，则创建
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 创建recorder，记录训练信息
    if (args.test):
        writer = SummaryWriter(f"{args.result_dir}/test_{args.project}__{args.timestamp}")
    else:
        writer = SummaryWriter(f"{args.result_dir}/train_{args.project}__{args.timestamp}")
    recorder = Recorder(args, writer, id2word)

    # 打印参数
    # for arg in vars(args):
    #  print (arg, getattr(args, arg))
    print(args)
    if not args.test:
        train(args, len(vocab))
    else:
        test(args, len(vocab))
