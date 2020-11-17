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
from sklearn import metrics
from models.model_k_party_chexpert import NetworkChexpert_A, NetworkChexpert_B
from dataset import MultiViewDataset, MultiViewDataset6Party, ChexpertDataset

parser = argparse.ArgumentParser("CheXpert-v1.0-small")
parser.add_argument('--data', type=str, default='data/CheXpert-v1.0-small',
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
parser.add_argument('--genotypes_file', type=str, required=True, help='which architecture_A to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--u_dim', type=int, default=64, help='u layer dimensions')
parser.add_argument('--k', type=int, required=True, help='num of clients')
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
    genotype_list = genotypes.parse_str(args.genotypes_file)
    assert len(genotype_list) == args.k, "len(genotype_list) must equal to k party."
    genotype_list = [eval("genotypes.%s" % genotype) for genotype in genotype_list]
    model_A = NetworkChexpert_A(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, genotype_list[0],
                                u_dim=args.u_dim, k=args.k)
    model_list = [model_A] + [
        NetworkChexpert_B(args.init_channels, args.layers, args.auxiliary, genotype_list[i], u_dim=args.u_dim) for i in
        range(1, args.k)]
    [model_list[i].cuda() for i in range(args.k)]
    for i in range(args.k):
        logging.info("model_{} param size = {}MB".format(i + 1, utils.count_parameters_in_MB(model_list[i])))

    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.cuda()
    optimizer_list = [
        torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        for model in model_list]

    train_data = ChexpertDataset(args.data, 'train', 224, 224, k=args.k)
    valid_data = ChexpertDataset(args.data, 'test', 224, 224, k=args.k)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    if args.learning_rate == 0.025:
        scheduler_list = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
            for optimizer in optimizer_list]
    else:
        scheduler_list = [torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma) for optimizer
                          in optimizer_list]

    best_auc = 0.
    best_acc = 0.
    for epoch in range(args.epochs):
        [scheduler_list[i].step() for i in range(len(scheduler_list))]
        lr = scheduler_list[0].get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        cur_step = epoch * len(train_queue)
        writer.add_scalar('train/lr', lr, cur_step)
        for i in range(args.k):
            model_list[i].drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model_list, optimizer_list, epoch)
        logging.info('train_acc %f', train_acc)

        cur_step = (epoch + 1) * len(train_queue)
        valid_acc, valid_auc, valid_obj = infer(valid_queue, model_list, epoch, cur_step)
        logging.info('valid_acc %f', valid_acc)
        logging.info('valid_auc %f', valid_auc)

        if best_acc < valid_acc:
            best_acc = valid_acc
        if best_auc < valid_auc:
            best_auc = valid_auc
    logging.info("Final best ACC = %f", best_acc)
    logging.info("Final best AUC = %f", best_auc)


def train(train_queue, model_list, optimizer_list, epoch):
    objs = utils.AvgrageMeter()
    acc = utils.AvgrageMeter()
    acc_sum = np.zeros(NUM_CLASSES)
    loss_sum = np.zeros(NUM_CLASSES)

    cur_step = epoch * len(train_queue)

    model_list = [model.train() for model in model_list]
    k = len(model_list)

    for step, (trn_X, trn_y) in enumerate(train_queue):
        trn_X = [x.float().cuda() for x in trn_X]
        target = trn_y.float().cuda()
        n = target.size(0)
        [optimizer_list[i].zero_grad() for i in range(k)]
        U_B_list = None
        if k > 1:
            U_B_list = [model_list[i](trn_X[i]) for i in range(1, len(model_list))]
        logits = model_list[0](trn_X[0], U_B_list)
        loss = 0
        for t in range(NUM_CLASSES):
            loss_t, acc_t, _ = utils.get_loss(logits, target, t)
            loss += loss_t
            loss_sum[t] += loss_t.item()
            acc_sum[t] += acc_t.item()
        objs.update(loss.item(), n)
        acc.update(sum(acc_sum) / NUM_CLASSES * 100, n)
        acc_sum = np.zeros(NUM_CLASSES)
        if k > 1:
            U_B_gradients_list = [torch.autograd.grad(loss, U_B, retain_graph=True) for U_B in U_B_list]
            model_B_weights_gradients_list = [
                torch.autograd.grad(U_B_list[i], model_list[i + 1].parameters(), grad_outputs=U_B_gradients_list[i],
                                    retain_graph=True) for i in range(len(U_B_gradients_list))]
            for i in range(len(model_B_weights_gradients_list)):
                for w, g in zip(model_list[i + 1].parameters(), model_B_weights_gradients_list[i]):
                    w.grad = g.detach()
                nn.utils.clip_grad_norm_(model_list[i + 1].parameters(), args.grad_clip)
                optimizer_list[i + 1].step()
        loss.backward()
        nn.utils.clip_grad_norm_(model_list[0].parameters(), args.grad_clip)
        optimizer_list[0].step()

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
    model_list = [model.eval() for model in model_list]
    k = len(model_list)
    pred_list = [[] for _ in range(NUM_CLASSES)]
    true_list = [[] for _ in range(NUM_CLASSES)]
    acc_sum = np.zeros(NUM_CLASSES)
    loss_sum = np.zeros(NUM_CLASSES)

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
