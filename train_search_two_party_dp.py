import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
from tensorboardX import SummaryWriter

from models.model_search_two_party import Network_A, Network_B
from architects.architect_two_party import Architect_A, Architect_B
from dataset import MultiViewDataset
from dp_utils import add_dp_v3

parser = argparse.ArgumentParser("modelnet_manually_aligned_png_full")
parser.add_argument('--data', type=str, default='data/modelnet_manually_aligned_png_full',
                    help='location of the data corpus')
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--u_dim', type=int, default=64, help='u layer dimensions')
parser.add_argument('--clip_norm_out', default=5.0, type=float)
parser.add_argument('--clip_norm_grad', default=0.2, type=float)
parser.add_argument('--noise_variance_out', default=1.0, type=float)
parser.add_argument('--noise_variance_grad', default=0.1, type=float)

args = parser.parse_args()

args.name = 'search/{}-{}'.format(args.name, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.name, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.name, 'log.txt'), mode='w')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(args.name, 'tb'))
writer.add_text('expername', args.name, 0)

NUM_CLASSES = 40


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

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model_A = Network_A(args.init_channels, NUM_CLASSES, args.layers, criterion, u_dim=args.u_dim)
    model_B = Network_B(args.init_channels, args.layers, criterion, u_dim=args.u_dim)
    model_A = model_A.cuda()
    model_B = model_B.cuda()
    logging.info("model_A param size = %fMB", utils.count_parameters_in_MB(model_A))
    logging.info("model_B param size = %fMB", utils.count_parameters_in_MB(model_B))

    optimizer_A = torch.optim.SGD(model_A.parameters(), args.learning_rate, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    optimizer_B = torch.optim.SGD(model_B.parameters(), args.learning_rate, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    train_data = MultiViewDataset(args.data, 'train', 32, 32)

    num_train = len(train_data)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=0)

    scheduler_A = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_A, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_B = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_B, float(args.epochs), eta_min=args.learning_rate_min)

    architect_A = Architect_A(model_A, args)
    architect_B = Architect_B(model_B, args)

    best_top1 = 0.
    for epoch in range(args.epochs):
        scheduler_A.step()
        scheduler_B.step()
        lr = scheduler_A.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype_A = model_A.genotype()
        genotype_B = model_B.genotype()
        logging.info('genotype_A = %s', genotype_A)
        logging.info('genotype_B = %s', genotype_B)
        logging.info('Model_A alphas')

        logging.info(F.softmax(model_A.alphas_normal, dim=-1))
        logging.info(F.softmax(model_B.alphas_reduce, dim=-1))
        logging.info('Model_B alphas')

        logging.info(F.softmax(model_B.alphas_normal, dim=-1))
        logging.info(F.softmax(model_B.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model_A, model_B, architect_A, architect_B, criterion,
                                     optimizer_A, optimizer_B, lr, epoch)
        logging.info('train_acc %f', train_acc)

        # validation
        cur_step = (epoch) * len(train_queue)
        valid_acc, valid_obj = infer(valid_queue, model_A, model_B, criterion, epoch, cur_step)
        logging.info('valid_acc %f', valid_acc)
        if best_top1 < valid_acc:
            best_top1 = valid_acc
            best_genotype_A = genotype_A
            best_genotype_B = genotype_B
        logging.info("Final best Prec@1 = {:.4%}".format(best_top1))
        logging.info("Best Genotype_A = {}".format(best_genotype_A))
        logging.info("Best Genotype_B = {}".format(best_genotype_B))

        utils.save(model_A, os.path.join(args.name, 'model_A_weights.pt'))
        utils.save(model_B, os.path.join(args.name, 'model_B_weights.pt'))


def train(train_queue, valid_queue, model_A, model_B, architect_A, architect_B, criterion, optimizer_A, optimizer_B, lr,
          epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    cur_step = epoch * len(train_queue)
    writer.add_scalar('train/lr', lr, cur_step)

    for step, (trn_X_A, trn_X_B, trn_y) in enumerate(train_queue):
        model_A.train()
        model_B.train()
        input_A = trn_X_A.float().cuda()
        input_B = trn_X_B.float().cuda()
        target = trn_y.view(-1).long().cuda()
        n = input_A.size(0)

        # get a random minibatch from the search queue with replacement
        (val_X_A, val_X_B, val_y) = next(iter(valid_queue))
        input_search_A = val_X_A.float().cuda()
        input_search_B = val_X_B.float().cuda()
        target_search = val_y.view(-1).long().cuda()

        U_B_val = model_B.get_u(input_search_B)

        ###### add noise to U_B_val
        U_B_val_noised = add_dp_v3(U_B_val, args.clip_norm_out, args.noise_variance_out)

        U_B_gradients = architect_A.update_alpha(input_search_A, U_B_val_noised, target_search)

        ##### add noise to U_B_gradients
        U_B_gradients_noised = add_dp_v3(U_B_gradients[0], args.clip_norm_grad, args.noise_variance_grad)

        architect_B.update_alpha(U_B_val, U_B_gradients_noised)

        U_B_train = model_B.get_u(input_B)

        ###### add noise to U_B_train
        U_B_train_noised = add_dp_v3(U_B_train, args.clip_norm_out, args.noise_variance_out)

        U_B_gradients, logits, loss = architect_A.update_weights(input_A, U_B_train_noised, target, optimizer_A,
                                                                 args.grad_clip)

        ##### add noise to U_B_gradients
        U_B_gradients_noised = add_dp_v3(U_B_gradients[0], args.clip_norm_grad, args.noise_variance_grad)

        architect_B.update_weights(U_B_train, U_B_gradients_noised, optimizer_B, args.grad_clip)

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
        for step, (val_X_A, val_X_B, val_y) in enumerate(valid_queue):
            input_A = val_X_A.float().cuda()
            input_B = val_X_B.float().cuda()
            target = val_y.view(-1).long().cuda()
            n = val_X_A.size(0)

            U_B = model_B(input_B)

            ###### add noise to U_B
            U_B_noised = add_dp_v3(U_B, args.clip_norm_out, args.noise_variance_out)

            loss, logits = model_A._loss(input_A, U_B_noised, target)

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
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
