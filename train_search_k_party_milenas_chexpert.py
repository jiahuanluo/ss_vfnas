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
import torch.backends.cudnn as cudnn
import random
from tensorboardX import SummaryWriter
from sklearn import metrics
from models.model_search_k_party_chexpert import Network_A, Network_B
from architects.architect_k_party_milenas import Architect_A, Architect_B
from dataset import MultiViewDataset, MultiViewDataset6Party, ChexpertDataset

parser = argparse.ArgumentParser("CheXpert-v1.0-small")
parser.add_argument('--data', type=str, default='data/CheXpert-v1.0-small',
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
parser.add_argument('--k', type=int, required=True, help='num of client')
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

NUM_CLASSES = 5


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

    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()
    model_A = Network_A(args.init_channels, NUM_CLASSES, args.layers, criterion, u_dim=args.u_dim, k=args.k)
    model_list = [model_A] + [Network_B(args.init_channels, args.layers, criterion, u_dim=args.u_dim) for _ in
                              range(args.k - 1)]
    model_list = [model.cuda() for model in model_list]

    optimizer_list = [
        torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        for model in model_list]
    # train_data = MultiViewDataset(args.data, 'train', 32, 32)
    train_data = ChexpertDataset(args.data, 'train', 32, 32, k=args.k)

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

    scheduler_list = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
        for optimizer in optimizer_list]

    architect_A = Architect_A(model_A, args)
    architect_list = [architect_A] + [Architect_B(model_list[i], args) for i in range(1, args.k)]

    best_auc = 0.
    best_acc = 0.
    for epoch in range(args.epochs):
        [scheduler_list[i].step() for i in range(args.k)]
        lr = scheduler_list[0].get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model_list, architect_list, optimizer_list,
                                     lr, epoch)
        logging.info('train_acc %f', train_acc)

        # validation
        cur_step = (epoch + 1) * len(train_queue)
        valid_acc, valid_auc, valid_obj = infer(valid_queue, model_list, epoch, cur_step)
        logging.info('valid_acc %f', valid_acc)
        logging.info('valid_auc %f', valid_auc)
        #
        for i in range(args.k):
            logging.info("Genotype_{} = {}".format(i + 1, model_list[i].genotype()))
        if best_acc < valid_acc:
            best_acc = valid_acc
        if best_auc < valid_auc:
            best_auc = valid_auc
            best_genotype_list = [model.genotype() for model in model_list]
    logging.info("Final best ACC = %f", best_acc)
    logging.info("Final best AUC = %f", best_auc)
    for i in range(args.k):
        logging.info("Best Genotype_{} = {}".format(i + 1, best_genotype_list[i]))


def train(train_queue, valid_queue, model_list, architect_list, optimizer_list, lr, epoch):
    objs = utils.AvgrageMeter()
    acc = utils.AvgrageMeter()
    acc_sum = np.zeros(NUM_CLASSES)
    loss_sum = np.zeros(NUM_CLASSES)

    cur_step = epoch * len(train_queue)
    writer.add_scalar('train/lr', lr, cur_step)
    model_list = [model.train() for model in model_list]
    k = len(model_list)
    for step, (trn_X, trn_y) in enumerate(train_queue):
        trn_X = [x.float().cuda() for x in trn_X]

        target = trn_y.float().cuda()
        n = target.size(0)

        # get a random minibatch from the search queue with replacement
        (val_X, val_y) = next(iter(valid_queue))
        val_X = [x.float().cuda() for x in val_X]
        target_search = val_y.float().cuda()
        U_B_train_list = None
        U_B_val_list = None
        if k > 1:
            U_B_train_list = [model_list[i].get_u(trn_X[i]) for i in range(1, len(trn_X))]
            U_B_val_list = [model_list[i].get_u(val_X[i]) for i in range(1, len(val_X))]
        U_B_train_gradients_list, train_alpha_gradients, train_weights_gradients, logits, loss = architect_list[
            0].compute_grad(trn_X[0], U_B_train_list, target, need_weight_grad=True)
        U_B_val_gradients_list, val_alpha_gradients = architect_list[0].compute_grad(val_X[0], U_B_val_list,
                                                                                     target_search,
                                                                                     need_weight_grad=False)
        architect_list[0].update(train_alpha_gradients, train_weights_gradients, val_alpha_gradients, optimizer_list[0],
                                 args.grad_clip)
        if k > 1:
            [architect_list[i + 1].update_weights(U_B_train_list[i], U_B_train_gradients_list[i], optimizer_list[i + 1],
                                                  args.grad_clip) for i in range(len(U_B_val_list))]
            [architect_list[i + 1].update_alpha(U_B_train_list[i], U_B_train_gradients_list[i], U_B_val_list[i],
                                                U_B_val_gradients_list[i]) for i in range(len(U_B_val_list))]
        optimizer_list[0].step()
        for t in range(NUM_CLASSES):
            loss_t, acc_t, _ = utils.get_loss(logits, target, t)
            loss += loss_t
            loss_sum[t] += loss_t.item()
            acc_sum[t] += acc_t.item()
        objs.update(loss.item(), n)
        acc.update(sum(acc_sum) / NUM_CLASSES * 100, n)
        acc_sum = np.zeros(NUM_CLASSES)

        if step % args.report_freq == 0:
            logging.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Acc ({top1.avg:.1f}%)".format(
                    epoch + 1, args.epochs, step, len(train_queue) - 1, losses=objs, top1=acc))
        writer.add_scalar('train/loss', objs.avg, cur_step)
        writer.add_scalar('train/acc', acc.avg, cur_step)
        cur_step += 1
    return acc.avg, objs.avg


def infer(valid_queue, model_list, epoch, cur_step):
    objs = utils.AvgrageMeter()
    acc = utils.AvgrageMeter()
    k = len(model_list)
    pred_list = [[] for _ in range(NUM_CLASSES)]
    true_list = [[] for _ in range(NUM_CLASSES)]
    acc_sum = np.zeros(NUM_CLASSES)
    loss_sum = np.zeros(NUM_CLASSES)
    [model.eval() for model in model_list]

    with torch.no_grad():
        for step, (val_X, val_y) in enumerate(valid_queue):
            val_X = [x.float().cuda() for x in val_X]
            target = val_y.float().cuda()
            n = target.size(0)
            U_B_list = None
            if k > 1:
                U_B_list = [model_list[i](val_X[i]) for i in range(1, len(model_list))]
            output = model_list[0](val_X[0], U_B_list)
            for t in range(NUM_CLASSES):
                loss_t, acc_t, _ = utils.get_loss(output, target, t)
                output_tensor = torch.sigmoid(output[t].view(-1)).cpu().detach().numpy()
                target_tensor = target[:, t].view(-1).cpu().detach().numpy()
                if step == 0:
                    pred_list[t] = output_tensor
                    true_list[t] = target_tensor
                else:
                    pred_list[t] = np.append(pred_list[t], output_tensor)
                    true_list[t] = np.append(true_list[t], target_tensor)
                loss_sum[t] += loss_t.item()
                acc_sum[t] += acc_t.item()
            objs.update(sum(loss_sum), n)
            acc.update(sum(acc_sum) / NUM_CLASSES * 100, n)
            acc_sum = np.zeros(NUM_CLASSES)
            loss_sum = np.zeros(NUM_CLASSES)
            if step % args.report_freq == 0:
                logging.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Acc ({top1.avg:.1f}%)".format(
                        epoch + 1, args.epochs, step, len(valid_queue) - 1, losses=objs,
                        top1=acc))
        auclist = []
        for i in range(NUM_CLASSES):
            y_pred = pred_list[i]
            y_true = true_list[i]
            fpr, tpr, thresholds = metrics.roc_curve(
                y_true, y_pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            auclist.append(auc)
        avg_auc = sum(auclist) / NUM_CLASSES * 100.
        logging.info(
            "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
            "Acc ({top1.avg:.1f}%, Auc {top5:.1f}%)".format(
                epoch + 1, args.epochs, step, len(valid_queue) - 1, losses=objs,
                top1=acc, top5=avg_auc))
        writer.add_scalar('valid/loss', objs.avg, cur_step)
        writer.add_scalar('valid/acc', acc.avg, cur_step)
        writer.add_scalar('valid/auc', avg_auc, cur_step)
        return acc.avg, avg_auc, objs.avg


if __name__ == '__main__':
    main()
