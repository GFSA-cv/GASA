import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  
import time
import shutil
import torch
import numpy
from torch.autograd import Variable
import logging
import tensorboard_logger as tb_logger
import argparse
import pickle

import data
from vocab import Vocabulary, deserialize_vocab
from evaluation import i2t_sep_sim, t2i_sep_sim, AverageMeter, LogCollector, encode_data
from model import GASA

torch.set_num_threads(10)


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./images/', help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp', help='{coco_precomp_original, f30k_precomp')
    parser.add_argument('--vocab_path', default='./vocab/', help='Path to saved vocabulary json files.')
    parser.add_argument('--orig_img_path', default='./data/', help='path to get the original image data')
    parser.add_argument('--orig_data_name', default='f30k', help='{coco,f30k}')
    parser.add_argument('--use_restval', action='store_false', help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=256, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=200, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=2000, type=int, help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/f30k/GASA_f30k/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/f30k/GASA_f30k/',
                        help='Path to save the model.')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_false', help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|weight_norm')
    parser.add_argument('--bi_gru', action='store_false', help='Use bidirectional GRU.')
    parser.add_argument('--use_BatchNorm', action='store_false', help='Whether to use BN.')
    parser.add_argument('--activation_type', default='tanh',
                        help='choose type of activation functions.')
    parser.add_argument('--dropout_rate', default=0.5, type=float,
                        help='dropout rate.')
    parser.add_argument('--use_abs', default='None', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--norm_func_type', default='sigmoid', help='choose type of norm functions.')
    parser.add_argument('--feature_fuse_type', default='weight_sum',
                        help='choose the fusing type for raw feature and attribute feature (multiple|concat|adap_sum|weight_sum))')
    parser.add_argument('--wemb_type', default='glove', choices=('glove', 'fasttext', 'random_init'), type=str,
                        help='Word embedding (glove|fasttext|random_init)')

    opt = parser.parse_args()
    print(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)
    '''load the vocab word2idx'''
    word2idx = vocab.word2idx

    # Load data loaders
    train_loader, val_loader = data.get_loaders(opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    # Construct the model
    # model = GASA(opt)
    model = GASA(word2idx, opt)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    for epoch in range(opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        rsum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        model.epoch = epoch

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model)


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    if 'coco' in opt.data_name:
        alpha = 0.9
    elif 'f30k' in opt.data_name:
        alpha = 0.85

    img_embs, cap_embs = encode_data(model=model, data_loader=val_loader, alpha=alpha)
    print(img_embs.shape[0] // 5, "Images", cap_embs.shape[0], "texts for validate")
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t_sep_sim(img_embs, cap_embs, opt.data_name,
                                             weight_fused=0.5,
                                             measure=opt.measure)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i_sep_sim(img_embs, cap_embs, opt.data_name,
                                                 weight_fused=0.5,
                                                 measure=opt.measure)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)

    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)

    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr_base = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr_base


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
