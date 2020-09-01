import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import genotypes
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from models.model_two_party import NetworkMulitview_A, NetworkMulitview_B
from dataset import MultiViewDataset, MultiViewDataset6Party

parser = argparse.ArgumentParser("modelnet_manually_aligned_png_full")
parser.add_argument('--data', type=str, default='data/modelnet_manually_aligned_png_full',
                    help='location of the data corpus')
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--genotypes_A', type=str,
                    default="Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))",
                    help='which architecture_A to use')
parser.add_argument('--genotypes_B', type=str,
                    default="Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6))",
                    help='which architecture_B to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--u_dim', type=int, default=64, help='u layer dimensions')
args = parser.parse_args()

args.name = 'eval/{}-{}'.format(args.name, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.name, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.name, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(args.name, 'tb'))
writer.add_text('expername', args.name, 0)

NUM_CLASSES = 40


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed_all(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype_A = eval("genotypes.%s" % args.genotypes_A)
    genotype_B = eval("genotypes.%s" % args.genotypes_B)
    model_A = NetworkMulitview_A(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, genotype_A,
                                 u_dim=args.u_dim)
    model_B = NetworkMulitview_B(args.init_channels, args.layers, args.auxiliary, genotype_B, u_dim=args.u_dim)
    model_A = model_A.cuda()
    model_B = model_B.cuda()

    logging.info("model_A param size = %fMB", utils.count_parameters_in_MB(model_A))
    logging.info("model_B param size = %fMB", utils.count_parameters_in_MB(model_B))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(NUM_CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer_A = torch.optim.SGD(
        model_A.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optimizer_B = torch.optim.SGD(
        model_B.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    # train_data = MultiViewDataset(args.data, 'train', 224, 224)
    # valid_data = MultiViewDataset(args.data, 'test', 224, 224)
    train_data = MultiViewDataset6Party(args.data, 'train', 224, 224, k=2)
    valid_data = MultiViewDataset6Party(args.data, 'test', 224, 224, k=2)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    if args.learning_rate == 0.025:
        scheduler_A = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_A, float(args.epochs))
        scheduler_B = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_B, float(args.epochs))
    else:
        scheduler_A = torch.optim.lr_scheduler.StepLR(optimizer_A, args.decay_period, gamma=args.gamma)
        scheduler_B = torch.optim.lr_scheduler.StepLR(optimizer_B, args.decay_period, gamma=args.gamma)

    best_acc_top1 = 0
    for epoch in range(args.epochs):
        scheduler_A.step()
        scheduler_B.step()
        lr = scheduler_A.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        cur_step = epoch * len(train_queue)
        writer.add_scalar('train/lr', lr, cur_step)
        model_A.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        model_B.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model_A, model_B, criterion_smooth, optimizer_A, optimizer_B, epoch)
        logging.info('train_acc %f', train_acc)

        cur_step = (epoch + 1) * len(train_queue)
        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model_A, model_B, criterion, epoch, cur_step)

        logging.info('valid_acc_top1 %f', valid_acc_top1)
        logging.info('valid_acc_top5 %f', valid_acc_top5)

        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        logging.info('best_acc_top1 %f', best_acc_top1)

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_A': model_A.state_dict(),
            'state_dict_B': model_B.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer_A': optimizer_A.state_dict(),
            'optimizer_B': optimizer_B.state_dict(),
        }, is_best, args.name)


def train(train_queue, model_A, model_B, criterion, optimizer_A, optimizer_B, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    cur_step = epoch * len(train_queue)

    model_A.train()
    model_B.train()

    for step, (trn_X, trn_y) in enumerate(train_queue):
        input_A = trn_X[0].float().cuda()
        input_B = trn_X[1].float().cuda()
        target = trn_y.view(-1).long().cuda()
        n = input_A.size(0)
        optimizer_A.zero_grad()
        U_B = model_B(input_B)
        logits = model_A(input_A, U_B)
        loss = criterion(logits, target)
        U_B_gradients = torch.autograd.grad(loss, U_B, retain_graph=True)
        model_B_weights_gradients = torch.autograd.grad(U_B, model_B.parameters(), grad_outputs=U_B_gradients,
                                                        retain_graph=True)
        for w, g in zip(model_B.parameters(), model_B_weights_gradients):
            w.grad = g.detach()
        nn.utils.clip_grad_norm_(model_B.parameters(), args.grad_clip)
        optimizer_B.step()
        loss.backward()
        nn.utils.clip_grad_norm_(model_A.parameters(), args.grad_clip)
        optimizer_A.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1f}%, {top5.avg:.1f}%)".format(
                    epoch + 1, args.epochs, step, len(train_queue) - 1, losses=objs,
                    top1=top1, top5=top5))
        writer.add_scalar('train/loss', objs.avg, cur_step)
        writer.add_scalar('train/top1', top1.avg, cur_step)
        writer.add_scalar('train/top5', top5.avg, cur_step)
        cur_step += 1
    return top1.avg, objs.avg


def infer(valid_queue, model_A, model_B, criterion, epoch, cur_step):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model_A.eval()
    model_B.eval()
    with torch.no_grad():
        for step, (val_X_A, val_X, val_y) in enumerate(valid_queue):
            input_A = val_X[0].float().cuda()
            input_B = val_X[1].float().cuda()
            target = val_y.view(-1).long().cuda()
            n = input_A.size(0)
            U_B = model_B(input_B)
            logits = model_A(input_A, U_B)
            loss = criterion(logits, target)
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1f}%, {top5.avg:.1f}%)".format(
                        epoch + 1, args.epochs, step, len(valid_queue) - 1, losses=objs,
                        top1=top1, top5=top5))
    writer.add_scalar('valid/loss', objs.avg, cur_step)
    writer.add_scalar('valid/top1', top1.avg, cur_step)
    writer.add_scalar('valid/top5', top5.avg, cur_step)
    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()
