import argparse
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import imagenet_network as models
import numpy as np

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision import transforms

from breeds_inc import BREEDSFactory

import os
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
import json
import logging

import pickle


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# lighting transform
# https://git.io/fhBOc
IMAGENET_PCA = {
    'eigval':torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec':torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}
class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))


def get_optimizer(optimizer_name, parameters, lr, momentum=0, weight_decay=0):
    if optimizer_name == 'sgd':
        return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'nesterov_sgd':
        return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def validate(val_loader, model, criterion, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output, _ = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader),
                                                                     batch_time=batch_time, loss=losses,
                                                                     top1=top1, top5=top5))
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def validate_with_new_old_model(val_loader, model, model_old, criterion, alpha, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model_old.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output, _ = model(input)
            output_old, _ = model_old(input)

            _, pred_old = output_old.topk(1, 1, True, True)
            pred_old = pred_old.t()
            #             print("Old Prediction: {}".format(pred_old[0]))

            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            #             print("New Prediction: {}".format(pred[0]))

            output_new = output_old + alpha * output

            _, pred_new = output_new.topk(1, 1, True, True)
            pred_new = pred_new.t()
            #             print("Combination Prediction: {}".format(pred_new[0]))

            loss = criterion(output_new, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output_new, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader),
                                                                     batch_time=batch_time, loss=losses,
                                                                     top1=top1, top5=top5))
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def alpha_evaluation_(temp_model,
                      temp_model_old,
                      train_val_loader,
                      criterion,
                      logger):

    performance_dict = dict()

    for alpha in np.arange(2.0, 0.0, -0.05):
        logger.info("alpha: {}".format(alpha))
        performance_dict[alpha] = dict()
        train_val_top1, _, _ = validate_with_new_old_model(train_val_loader, temp_model, temp_model_old, criterion,alpha, logger)
        performance_dict[alpha]['train_val_top1'] = train_val_top1.cpu().item()

        logger.info("\n")

    alpha = 0
    logger.info("alpha: {}".format(alpha))
    performance_dict[alpha] = dict()
    train_val_top1, _, _ = validate_with_new_old_model(train_val_loader, temp_model, temp_model_old, criterion, alpha, logger)
    performance_dict[alpha]['train_val_top1'] = train_val_top1.cpu().item()

    return performance_dict


def parse_args():
    parser = argparse.ArgumentParser(description='train ISL')
    # general
    parser.add_argument('--ds_name',
                        help='dataset name',
                        required=True,
                        type=str)
    parser.add_argument('--inc_step_num',
                        help='incremental steps size',
                        required=True,
                        type=int)
    parser.add_argument('--info_dir',
                        help='breeds benchmark info path',
                        required=False,
                        type=str,
                        default='/root/autodl-tmp/BREEDS-Benchmarks/imagenet_class_hierarchy/modified')
    parser.add_argument('--data_dir',
                        help='data path',
                        required=False,
                        type=str,
                        default='/root/autodl-tmp/ILSVRC2012_Data')
    parser.add_argument('--base_step_pretrained_path',
                        help='base step pretrained model path',
                        required=False,
                        type=str,
                        default='ckpts/test_breeds_entity_30_standard_data_augment_true_300_Epoch_true_step_100_epoch_bs_128_resnet_18/fbresnet18/model_best.pth.tar')
    parser.add_argument('--task_stat_path',
                        help='experiment configure file name',
                        required=False,
                        type=str,
                        default='experiments/entity30_15_tasks.pkl')
    parser.add_argument('--exp_name',
                        help='experiment name',
                        required=False,
                        type=str,
                        default='debug_15_task_lr_5e-3_wd1e-4_mo9e-1_test')
    parser.add_argument('--retrain_epoch',
                        help='incremental step training epoch',
                        required=False,
                        type=int,
                        default=20)
    parser.add_argument('--IL_initial_LR',
                        help='incremental learning rate',
                        required=False,
                        type=float,
                        default=0.005)
    parser.add_argument('--wd',
                        help='weigth decay',
                        required=False,
                        type=float,
                        default=0.0001)
    parser.add_argument('--mo',
                        help='momentum',
                        required=False,
                        type=float,
                        default=0.9)


    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    log_path = 'logs/'
    os.makedirs(log_path, exist_ok=True)
    log_name = "{}.log".format(args.exp_name)
    logger = get_logger(log_path + log_name)

    ds_name = args.ds_name
    print("ds_name: {}".format(ds_name))

    breeds_factory = BREEDSFactory(info_dir=args.info_dir,
                                   data_dir=args.data_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    ''' create the source_train_val_augment_dataset to obtain the step-0's feature mean  '''
    source_train_val_augment_dataset = breeds_factory.get_breeds(
        ds_name=ds_name,
        partition='train',
        source=True,
        mode='coarse',
        transforms=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        split='rand'
    )

    logger.info('=> source_train_val_augment_dataset_size: {}'.format(len(source_train_val_augment_dataset)))
    logger.info('=> source_train_val_augment_dataset_number_of_class :{}'.format(source_train_val_augment_dataset))

    source_train_val_augment_dataset_loader = torch.utils.data.DataLoader(
        source_train_val_augment_dataset,
        batch_size=128, shuffle=False,
        num_workers=16, pin_memory=True)

    ''' create target_train dataset '''
    target_train_dataset = breeds_factory.get_breeds(
        ds_name=ds_name,
        partition='train',
        source=False,
        mode='coarse',
        transforms=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1),
            transforms.ToTensor(),
            Lighting(0.05, IMAGENET_PCA['eigval'],
                     IMAGENET_PCA['eigvec']),
            normalize,
        ]),
        split='rand'
    )

    logger.info('=> target_train_dataset_size: {}'.format(len(target_train_dataset)))
    logger.info('=> target_train_dataset_number_of_class :{}'.format(target_train_dataset.num_classes))

    ''' create target_train_val dataset '''
    target_train_val_augment_dataset = breeds_factory.get_breeds(
        ds_name=ds_name,
        partition='train',
        source=False,
        mode='coarse',
        transforms=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        split='rand'
    )

    logger.info('=> target_train_val_augment_dataset_size: {}'.format(len(target_train_val_augment_dataset)))
    logger.info('=> target_train_val_augment_dataset_number_of_class :{}'.format(target_train_val_augment_dataset))

    ''' create target_val dataset (i.e., test set) '''
    val_val_dataset = breeds_factory.get_breeds(
        ds_name=ds_name,
        partition='val',
        source=False,
        mode='coarse',
        transforms=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        split='rand'
    )

    logger.info('=> val_val_dataset_size: {}'.format(len(val_val_dataset)))
    logger.info('=> val_val_dataset_number_of_class :{}'.format(val_val_dataset.num_classes))

    ''' create the data loader for the whole test set for all the incremental steps '''
    val_val_loader = torch.utils.data.DataLoader(
        val_val_dataset,
        batch_size=128, shuffle=False,
        num_workers=16, pin_memory=True)

    ''' create the source_val dataset, i.e., step-0's test set '''
    val_source_val_dataset = breeds_factory.get_breeds(
        ds_name=ds_name,
        partition='val',
        source=True,
        mode='coarse',
        transforms=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        split='rand'
    )

    logger.info('=> val_source_val_dataset_size: {}'.format(len(val_source_val_dataset)))
    logger.info('=> val_source_val_dataset_number_of_class :{}'.format(val_source_val_dataset.num_classes))

    val_source_val_loader = torch.utils.data.DataLoader(
        val_source_val_dataset,
        batch_size=128, shuffle=False,
        num_workers=16, pin_memory=True)

    # print for debug
    logger.info("target_train_dataset.class_to_idx: {}".format(target_train_dataset.class_to_idx))
    logger.info("target_train_dataset.coarse2fine: {}".format(target_train_dataset.coarse2fine))
    logger.info("len(target_train_dataset.samples): {}".format(len(target_train_dataset.samples)))
    logger.info("target_train_dataset.class_to_idx.keys(): {}".format(target_train_dataset.class_to_idx.keys()))

    if ds_name == 'entity30':
        class_number = 30
    elif ds_name == 'entity13':
        class_number = 13

    logger.info("=> class_number: {}".format(class_number))

    coarse_to_fine_map = dict()
    for i in range(0, class_number):
        coarse_to_fine_map[i] = list()

    for key in target_train_dataset.class_to_idx.keys():
        coarse_to_fine_map[target_train_dataset.class_to_idx[key]].append(key)

    logger.info("=> coarse_to_fine_map: {}".format(coarse_to_fine_map))

    inc_step_num = args.inc_step_num
    logger.info("=> inc_step_num: {}".format(inc_step_num))

    task_coarse_class_dict = dict()
    task_size = inc_step_num + 1  # index start from 1
    logger.info("=> total task_size: {}".format(task_size))
    for i in range(1, task_size):
        task_coarse_class_dict[i] = list()

    """ 
    create the subclasses list for each step.
    this can be different for each protocols
    """
    task_stat_path = args.task_stat_path  # 'entity30_15_tasks.pkl'
    with open(task_stat_path, 'rb') as f:
        task_coarse_class_dict = pickle.load(f)

    logger.info("=> task_coarse_class_dict[1]: {}".format(task_coarse_class_dict[1]))
    # print("=> set(target_train_dataset.class_to_idx.keys()): {}".format(set(target_train_dataset.class_to_idx.keys())))

    ''' check the subclasses separation, different task may have different code '''
    logger.info("=> subclass intersection over each step: {}".format(
        set(task_coarse_class_dict[1]) & set(task_coarse_class_dict[2]) & set(task_coarse_class_dict[3]) & set(task_coarse_class_dict[4]) &
        set(task_coarse_class_dict[5]) & set(task_coarse_class_dict[6]) & set(task_coarse_class_dict[7]) & set(task_coarse_class_dict[8]) &
        set(task_coarse_class_dict[9]) & set(task_coarse_class_dict[10]) & set(task_coarse_class_dict[11]) & set(task_coarse_class_dict[12]) &
        set(task_coarse_class_dict[13]) & set(task_coarse_class_dict[14]) & set(task_coarse_class_dict[15])
        )
    )

    assert set.union(set(task_coarse_class_dict[1]), set(task_coarse_class_dict[2]), set(task_coarse_class_dict[3]),
                     set(task_coarse_class_dict[4]), set(task_coarse_class_dict[5]), set(task_coarse_class_dict[6]),
                     set(task_coarse_class_dict[7]), set(task_coarse_class_dict[8]), set(task_coarse_class_dict[9]),
                     set(task_coarse_class_dict[10]), set(task_coarse_class_dict[11]), set(task_coarse_class_dict[12]),
                     set(task_coarse_class_dict[13]), set(task_coarse_class_dict[14]), set(task_coarse_class_dict[15])) == set(target_train_dataset.class_to_idx.keys())

    logger.info("=> task_coarse_class_dict: {}".format(task_coarse_class_dict))

    '''
    create the train, train_val, test set and corresponding loaders
    '''

    ''' create each step's training images index dict '''
    task_training_idx_list_dict = dict()
    for i in range(1, task_size):
        logger.info("task {}".format(i))
        task_training_idx_list_dict[i] = list()
        for subclass in task_coarse_class_dict[i]:
            temp_list = [j for j in range(0, len(target_train_dataset.samples)) if
                         target_train_dataset.samples[j][2] == subclass]
            task_training_idx_list_dict[i].extend(temp_list)

    ''' create each step's training Subset dict '''
    dset_train_train_task_dict = dict()
    for i in range(1, task_size):
        dset_train_train_task_dict[i] = torch.utils.data.dataset.Subset(target_train_dataset,
                                                                        task_training_idx_list_dict[i])
        logger.info(len(dset_train_train_task_dict[i]))

    ''' create each step's training loader dict '''
    target_train_train_task_loader_dict = dict()
    for i in range(1, task_size):
        train_sampler = None
        target_train_train_task_loader_dict[i] = torch.utils.data.DataLoader(dset_train_train_task_dict[i],
                                                                             batch_size=128,
                                                                             shuffle=True,
                                                                             num_workers=16,
                                                                             pin_memory=True,
                                                                             sampler=train_sampler)

    ''' create each step's testing images index dict '''
    task_val_idx_list_dict = dict()
    for i in range(1, task_size):
        logger.info("task {}".format(i))
        task_val_idx_list_dict[i] = list()
        for subclass in task_coarse_class_dict[i]:
            temp_list = [j for j in range(0, len(val_val_dataset.samples)) if val_val_dataset.samples[j][2] == subclass]
            task_val_idx_list_dict[i].extend(temp_list)

    for i in range(1, task_size):
        logger.info("task {}, data size: {}".format(i, len(task_val_idx_list_dict[i])))

    ''' create each step's testing Subset dict '''
    dset_val_val_task_dict = dict()
    for i in range(1, task_size):
        dset_val_val_task_dict[i] = torch.utils.data.dataset.Subset(val_val_dataset, task_val_idx_list_dict[i])
        logger.info(len(dset_val_val_task_dict[i]))

    ''' create each step's testing loader dict'''
    target_val_val_task_loader_dict = dict()
    for i in range(1, task_size):
        target_val_val_task_loader_dict[i] = torch.utils.data.DataLoader(dset_val_val_task_dict[i],
                                                                         batch_size=128,
                                                                         shuffle=False,
                                                                         num_workers=16,
                                                                         pin_memory=True)

    ''' Create the target_train dataset using the val augmentation. 
    This is used for calculate the mean feature in each previous step '''

    task_target_train_val_augment_idx_list_dict = dict()
    for i in range(1, task_size):
        logger.info("task {}".format(i))
        task_target_train_val_augment_idx_list_dict[i] = list()
        for subclass in task_coarse_class_dict[i]:
            temp_list = [j for j in range(0, len(target_train_val_augment_dataset.samples)) if
                         target_train_val_augment_dataset.samples[j][2] == subclass]
            task_target_train_val_augment_idx_list_dict[i].extend(temp_list)

    for i in range(1, task_size):
        logger.info("=> task {}, data size: {}".format(i, len(task_target_train_val_augment_idx_list_dict[i])))

    dset_target_train_val_augment_task_dict = dict()
    for i in range(1, task_size):
        dset_target_train_val_augment_task_dict[i] = torch.utils.data.dataset.Subset(target_train_val_augment_dataset,
                                                                                     task_target_train_val_augment_idx_list_dict[i])
        logger.info(len(dset_target_train_val_augment_task_dict[i]))

    target_train_val_augment_task_loader_dict = dict()
    for i in range(1, task_size):
        train_sampler = None
        target_train_val_augment_task_loader_dict[i] = torch.utils.data.DataLoader(
            dset_target_train_val_augment_task_dict[i], batch_size=128, shuffle=False,
            num_workers=16, pin_memory=True)


    ''' create each step's train_val images index dict '''
    train_val_class_size = 50
    task_training_val_idx_list_dict = dict()
    for i in range(1, task_size):
        logger.info("task {}".format(i))
        task_training_val_idx_list_dict[i] = list()
        for subclass in task_coarse_class_dict[i]:
            temp_list = [j for j in range(0, len(target_train_val_augment_dataset.samples)) if
                         target_train_val_augment_dataset.samples[j][2] == subclass]
            task_training_val_idx_list_dict[i].extend(temp_list[0:train_val_class_size])

    ''' create each step's train_val Subset dict '''
    dset_train_train_val_task_dict = dict()
    for i in range(1, task_size):
        dset_train_train_val_task_dict[i] = torch.utils.data.dataset.Subset(target_train_val_augment_dataset,
                                                                            task_training_val_idx_list_dict[i])
        logger.info(len(dset_train_train_val_task_dict[i]))

    target_train_val_task_loader_dict = dict()
    for i in range(1, task_size):
        target_train_val_task_loader_dict[i] = torch.utils.data.DataLoader(dset_train_train_val_task_dict[i],
                                                                           batch_size=128,
                                                                           shuffle=False,
                                                                           num_workers=16,
                                                                           pin_memory=True)

    ''' training related things '''

    criterion = nn.CrossEntropyLoss().cuda()
    IL_initial_lr = args.IL_initial_LR
    logger.info("IL initial LR: {}".format(IL_initial_lr))
    retrain_epoch = args.retrain_epoch
    logger.info("retrain_epoch: {}".format(retrain_epoch))

    arch = 'fbresnet_extract_feature_18'
    model_old = models.__dict__[arch](num_classes=val_val_dataset.num_classes, pretrained=False)
    logger.info('=>    Total params: %.2fM' % (sum(p.numel() for p in model_old.parameters()) / 1000000.0))

    for par in model_old.parameters():
        par.requires_grad = False
    model_old.eval()
    model_old = nn.DataParallel(model_old).cuda()
    pretrain_model_path = args.base_step_pretrained_path
    checkpoint = torch.load(pretrain_model_path)
    model_old.load_state_dict(checkpoint['state_dict'])

    model = models.__dict__[arch](num_classes=target_train_dataset.num_classes, pretrained=False)
    logger.info('=>    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    model = nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])

    model_dict = dict()
    incremental_learning_momentum = args.mo
    logger.info("incremental_learning_momentum: {}".format(incremental_learning_momentum))
    incremental_learning_wd = args.wd
    logger.info("incremental_learning_wd: {}".format(incremental_learning_wd))
    load_best_train_val_model = True

    logger.info("=> start incremental learning!")
    for task_index in range(1, task_size):
        if task_index == 1:
            logger.info("=> initialize all_task_feature_label_dict ...")
            all_task_feature_label_dict = dict()
        else:
            logger.info("=> all_task_feature_label_dict already exists ...")

        previous_task_index = task_index - 1
        logger.info("=> previous_task_index: {}".format(previous_task_index))
        logger.info("=> before training, first obtain last step's feature mean ...")
        all_task_feature_label_dict, previous_feature_label_dict = calculate_last_step_feature(previous_task_index,
                                                                                               model_old,
                                                                                               target_train_val_augment_task_loader_dict,
                                                                                               all_task_feature_label_dict,
                                                                                               class_number,
                                                                                               source_train_val_augment_dataset_loader)

        # incremental training
        logger.info("=> start training on Inc Step {}".format(task_index))
        model, best_model_state, best_acc1 = inc_trainer(model,
                                                         model_old,
                                                         task_index,
                                                         target_train_train_task_loader_dict,
                                                         target_val_val_task_loader_dict,
                                                         target_train_val_task_loader_dict,
                                                         class_number,
                                                         IL_initial_lr,
                                                         retrain_epoch,
                                                         criterion,
                                                         incremental_learning_momentum,
                                                         incremental_learning_wd,
                                                         logger)

        # obtain the current step's train_val and val loader

        val_loader = target_val_val_task_loader_dict[task_index]
        train_val_loader = target_train_val_task_loader_dict[task_index]

        # load the best_model
        if load_best_train_val_model:
            logger.info("=> loading the best model ...")
            logger.info("=> evaluate on the Stage-1's model ...")
            model.load_state_dict(best_model_state)
            validate(val_loader, model, criterion, logger)
            validate(train_val_loader, model, criterion, logger)
            logger.info("=> evaluation completed ...")

        # obtain the performance dictionary
        logger.info("=> calculate performance dictionary ...")
        performance_dict = alpha_evaluation_(model,
                                             model_old,
                                             train_val_loader,
                                             criterion,
                                             logger
                                             )
        logger.info("=> completed performance dictionary ...")

        logger.info("=> calculate calculate_val_top1_list ...")
        train_val_top1_list = calculate_val_top1_list(performance_dict)
        logger.info("=> completed calculate_val_top1_list ...")

        if task_index == 1:
            logger.info("=> initialize num_of_task_dict in Inc Step {}".format(task_index))
            num_of_task_dict = dict()
        else:
            logger.info("=> num_of_task_dict already exists in Inc Step {}.".format(task_index))

        logger.info("=> get to know what classes are introduced new subclass in Inc Step {}.".format(task_index))
        num_of_task_dict = calculate_num_of_cls(previous_task_index, previous_feature_label_dict, num_of_task_dict, class_number)

        logger.info("=> calculate previous_tasks_alpha_dict...")
        previous_tasks_alpha_dict = get_each_previous_task_alpha_dict(model, model_old, task_index, all_task_feature_label_dict, num_of_task_dict)

        logger.info("=> calculate delta_dist_dict_temp ...")
        delta_dist_dict_temp = calculate_delta_dist_dict(previous_tasks_alpha_dict)

        logger.info("=> calculate gradient_ratio_list ...")
        gradient_ratio_list, top1_delta_large, delta_dist_large = calculate_graident_ratio_list(delta_dist_dict_temp, previous_tasks_alpha_dict, train_val_top1_list)
        logger.info("top1_delta_large: {}".format(top1_delta_large))

        alpha_list_len = len(np.arange(2.0, 0.0, -0.05)) # discretize the alpha value from [0, 2] with interval 0.5
        if top1_delta_large >= alpha_list_len // 2:
            logger.info("top1_delta mostly larger than delta_dist_list")
        else:
            logger.info("delta_dist_list larger than top1_delta")

        balanced_ratio = calculate_balanced_ratio(gradient_ratio_list)
        logger.info("balanced_ratio {} for Inc Step {}".format(balanced_ratio, task_index))
        best_alpha = calculate_best_alpha_2(delta_dist_dict_temp,
                                            previous_tasks_alpha_dict,
                                            train_val_top1_list,
                                            top1_delta_large,
                                            balanced_ratio)

        logger.info("Inc Step {}, best_alpha: {}".format(task_index, best_alpha))

        logger.info("=> perform Linear Combination after Inc Step {} Training".format(task_index))
        model = linear_combination(deepcopy(model_old), model, model_old, best_alpha, logger)
        logger.info("=> validate on model {}".format(task_index))
        validate(val_loader, model, criterion, logger)
        validate(val_val_loader, model, criterion, logger)
        validate(val_source_val_loader, model, criterion, logger)

        # For the new step, the model need to be trainable
        for par in model.parameters():
            par.requires_grad = True

        # for the new step, the model_old need to be eval
        model_old = deepcopy(model)
        for par in model_old.parameters():
            par.requires_grad = False
        model_old.eval()

        # re-initialize for next step
        logger.info("=> reinitialize the model and model_old for next Inc Step")

        logger.info("=> after Inc Step {}, validate on model_old".format(task_index))
        validate(val_loader, model_old, criterion, logger)
        validate(val_val_loader, model_old, criterion, logger)
        validate(val_source_val_loader, model_old, criterion, logger)

        model_dict[task_index] = deepcopy(model)
        path_name = 'incremental_ckpts/{}/task_{}'.format(args.exp_name, task_index)
        os.makedirs(path_name + '/' + arch, exist_ok=True)
        save_name = path_name + '/' + arch
        is_best = True
        save_checkpoint({
            'epoch': retrain_epoch,
            'arch': arch,
            'state_dict': model_old.state_dict(),
            'performance_dict': performance_dict,
        }, is_best, filename=save_name, epoch=retrain_epoch)

    logger.info("=> start calculating the final metrics...")
    task_performance = dict()
    for task_idx in range(1, task_size):
        task_performance[task_idx] = dict()

    for task_index in range(1, task_size):
        task_performance = per_task_performance(task_performance, model_dict[task_index], task_index, val_val_loader, val_source_val_loader,
                                                target_val_val_task_loader_dict, criterion, logger)

    average_forgetting = dict()
    previous_task_performance_dict = dict()

    for task_ind in range(0, task_size - 1):
        previous_task_performance_dict[task_ind] = list()

    for task_ind in task_performance.keys():
        for previous_task_ind in range(0, task_ind):
            print(previous_task_ind)
            previous_task_performance_dict[previous_task_ind].append(task_performance[task_ind][previous_task_ind])

    task_0_test_size = 6000
    each_task_test_size = 400 # 4 task: 1500, 8 tasks: 750, 15 tasks: 400
    average_top1 = dict()
    target_top1 = dict()

    for task_ind in task_performance.keys():
        denom = task_0_test_size + task_ind * each_task_test_size
        temp_acc = 0
        target_acc = 0
        temp_acc += task_performance[task_ind][0] * task_0_test_size / denom

        for previous_task_ind in range(1, task_ind):
            temp_acc += task_performance[task_ind][previous_task_ind] * each_task_test_size / denom
            target_acc += task_performance[task_ind][previous_task_ind]

        temp_acc += task_performance[task_ind]['current_task_val_top1'] * each_task_test_size / denom
        target_acc += task_performance[task_ind]['current_task_val_top1']
        target_top1[task_ind] = target_acc / task_ind
        average_top1[task_ind] = temp_acc

    logger.info("=> average_top1: {}".format(average_top1))
    logger.info("=> target_top1: {}".format(target_top1))

    result_dict = dict()
    result_dict['task_performance'] = task_performance
    result_dict['target_top1'] = target_top1
    result_dict['average_top1'] = average_top1

    path = 'results/{}_Tasks/{}/Ours/'.format(args.inc_step_num,args.exp_name)
    os.makedirs(path, exist_ok=True)

    result_file_name = '{}.json'.format(args.exp_name)
    with open(path + result_file_name, 'w') as fp:
        json.dump(result_dict, fp)


def per_task_performance(task_performance_dict,
                         temp_model,
                         task_ind,
                         val_val_loader,
                         val_source_val_loader,
                         target_val_val_task_loader_dict,
                         criterion,
                         logger):
    previous_task_list = sorted([i for i in range(1, task_ind)], reverse=True)
    val_loader = target_val_val_task_loader_dict[task_ind]
    current_task_val_top1, _, _ = validate(val_loader, temp_model, criterion, logger)
    task_performance_dict[task_ind]["current_task_val_top1"] = current_task_val_top1.cpu().item()

    if task_ind > 1:
        for previous_task_index in previous_task_list:
            print("previous task index: {}".format(previous_task_index))
            top1_acc_previous_task, _, _ = validate(target_val_val_task_loader_dict[previous_task_index], temp_model,
                                                    criterion, logger)
            task_performance_dict[task_ind][previous_task_index] = top1_acc_previous_task.cpu().item()

    top1_acc_val_all_task, _, _ = validate(val_val_loader, temp_model, criterion,logger)
    task_performance_dict[task_ind]["all_target_tasks"] = top1_acc_val_all_task.cpu().item()

    top1_acc_task_0, _, _ = validate(val_source_val_loader, temp_model, criterion,logger)
    task_performance_dict[task_ind][0] = top1_acc_task_0.cpu().item()

    return task_performance_dict

def save_checkpoint(state, is_best, filename, epoch):
    if epoch in [50-1]:
        torch.save(state, filename + '/checkpoint'+str(epoch)+'.pth.tar')
    torch.save(state, filename +'/checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename +'/checkpoint.pth.tar', filename + '/model_best.pth.tar')


def linear_combination(temp_model_new, temp_model, temp_model_old, alpha, logger):
    temp_model_new = temp_model_new.to('cpu')
    temp_model_new_state_dict = temp_model_new.state_dict()

    temp_model_old = temp_model_old.to('cpu')
    temp_model_old_state_dict = temp_model_old.state_dict()

    temp_model = temp_model.to('cpu')
    temp_model_state_dict = temp_model.state_dict()

    logger.info("best_alpha: {}".format(alpha))
    temp_model_new_state_dict['module.last_linear.weight'] = alpha * temp_model_state_dict['module.last_linear.weight'] + \
                                                             temp_model_old_state_dict['module.last_linear.weight']
    logger.info(temp_model_new_state_dict['module.last_linear.weight'])

    temp_model_new_state_dict['module.last_linear.bias'] = alpha * temp_model_state_dict['module.last_linear.bias'] + \
                                                           temp_model_old_state_dict['module.last_linear.bias']
    logger.info(temp_model_new_state_dict['module.last_linear.bias'])

    temp_model_new.load_state_dict(temp_model_new_state_dict)
    temp_model_new = temp_model_new.cuda()
    logger.info(temp_model_new.state_dict()['module.last_linear.weight'])

    temp_model_old = temp_model_old.cuda()
    temp_model = temp_model.cuda()

    return temp_model_new


def calculate_best_alpha_2(delta_dist_dict, previous_tasks_alpha_dict, val_top1_list, top1_delta_large, balanced_ratio):
    best_alpha = 0
    loss_list = list()

    for key in range(20, len(delta_dist_dict[0])):
        print(key)
        alpha = list(previous_tasks_alpha_dict[0].keys())[key]
        print("alpha: {}".format(alpha))

        top1_delta = val_top1_list[key] - val_top1_list[len(delta_dist_dict[0]) - 1]
        print("top1 delta: {}".format(top1_delta))

        forgetting_loss = list()
        for task_id in previous_tasks_alpha_dict.keys():
            temp_task_loss = delta_dist_dict[task_id][key] - delta_dist_dict[task_id][-1]
            print("task {} delta_dist_delta: {}".format(task_id, temp_task_loss))
            forgetting_loss.append(abs(temp_task_loss))

        if top1_delta_large > 10:  # top1 loss term is much larger
            if balanced_ratio < 0.5:
                loss = balanced_ratio * top1_delta - (1 - balanced_ratio) * sum(forgetting_loss)
            else:
                loss = (1 - balanced_ratio) * top1_delta - balanced_ratio * sum(forgetting_loss)
        elif abs(top1_delta_large - 10) <= 3:
            loss = top1_delta - sum(forgetting_loss)
        else:  # delta_dist_delta loss term is much larger
            if balanced_ratio < 0.5:
                loss = (1 - balanced_ratio) * top1_delta - (balanced_ratio) * sum(forgetting_loss)
            else:
                loss = balanced_ratio * top1_delta - (1 - balanced_ratio) * sum(forgetting_loss)

        print("alpha: {}, loss: {}".format(alpha, loss))
        loss_list.append(loss)
        if loss >= max(loss_list) and alpha != 0:
            best_alpha = alpha
    print("best alpha: {}".format(best_alpha))

    return best_alpha


def calculate_balanced_ratio(gradient_ratio_list):
    temp_mean = sum(gradient_ratio_list) / len(gradient_ratio_list)
    temp_min = min(gradient_ratio_list)

    if temp_mean - temp_min >= 0.5:
        balanced_ratio = temp_min
    else:
        if temp_min >= 0.5:  # all the gradient is very large, now it is safe to use the min to balanced two term
            balanced_ratio = temp_min
        elif temp_min >= 0.25:
            balanced_ratio = temp_min
        elif temp_mean >= 0.45 or temp_mean <= 0.55:  # it is much stable to use the mean of temp_mean and temp_min when the temp_mean is 0.5+-0.05
            balanced_ratio = (temp_min + temp_mean) / 2
        else:
            balanced_ratio = temp_mean

    return balanced_ratio


def calculate_graident_ratio_list(delta_dist_dict, previous_tasks_alpha_dict, val_top1_list):
    gradient_ratio_list = list()
    top1_delta_large = 0
    delta_dist_large = 0

    for key in range(20, len(delta_dist_dict[0])):
        print(key)
        alpha = list(previous_tasks_alpha_dict[0].keys())[key]

        top1_delta = val_top1_list[key] - val_top1_list[len(delta_dist_dict[0])-1]
        print("top1 delta: {}".format(top1_delta))

        forgetting_loss = list()
        for task_id in previous_tasks_alpha_dict.keys():
            temp_task_loss = delta_dist_dict[task_id][key] - delta_dist_dict[task_id][-1]
            print("task {} delta_dist_delta: {}".format(task_id, temp_task_loss))
            forgetting_loss.append(abs(temp_task_loss))

        if alpha != 0:
            if abs(top1_delta) > sum(forgetting_loss):
                gradient_ratio = sum(forgetting_loss) / abs(top1_delta)
                print('top1_delta > delta_dist_delta')
                top1_delta_large += 1
            else:
                gradient_ratio = abs(top1_delta) / sum(forgetting_loss)
                print('top1_delta < delta_dist_delta')
                delta_dist_large += 1
            gradient_ratio_list.append(gradient_ratio)

    print("gradient_ratio mean: {}".format(sum(gradient_ratio_list)/len(gradient_ratio_list)))
    print("gradient_ratio min: {}".format(min(gradient_ratio_list)))
    return gradient_ratio_list, top1_delta_large, delta_dist_large


def calculate_delta_dist_dict(temp_previous_tasks_alpha_dict):
    temp_delta_dist_dict = dict()

    for task_id in temp_previous_tasks_alpha_dict.keys():
        print("task_id: {}".format(task_id))
        temp_delta_list = list()
        temp_delta_dist_dict[task_id] = list()
        for alpha in temp_previous_tasks_alpha_dict[task_id].keys():
            print("alpha: {}".format(alpha))
            temp_alpha_sum_avg = sum(temp_previous_tasks_alpha_dict[task_id][alpha]) / len(
                temp_previous_tasks_alpha_dict[task_id][alpha])
            temp_delta_dist_dict[task_id].append(temp_alpha_sum_avg)
    return temp_delta_dist_dict


def calculate_feature_decision_boundary_distance(temp_feature_label_dict,
                                                 temp_model,
                                                 temp_model_old,
                                                 temp_alpha_dict,
                                                 num_of_cls):
    for key in temp_feature_label_dict.keys():
        if key not in num_of_cls:
            continue
        else:
            # print("class: {}".format(key))

            temp_features = np.array(temp_feature_label_dict[key])
            temp_feature_mean = np.mean(temp_features, axis=0)  # mean feature
            temp_feature_mean_torch = torch.from_numpy(temp_feature_mean).cuda(
                non_blocking=True)  # move to GPU for computation

            classify_score_model_old = temp_feature_mean_torch.matmul(
                temp_model_old.state_dict()['module.last_linear.weight'].t()) + temp_model_old.state_dict()[
                                           'module.last_linear.bias']
            # print("classify_score_model_old.shape[0]: {}".format(classify_score_model_old.shape[0]))
            # print("model_old classification score: {}".format(classify_score_model_old))
            _, indices_model_old = torch.topk(classify_score_model_old, 1)
            # print("model_old classification: {}".format(indices_model_old.t()))

            classify_score_model_new = temp_feature_mean_torch.matmul(
                temp_model.state_dict()['module.last_linear.weight'].t()) + temp_model.state_dict()[
                                           'module.last_linear.bias']
            # print("model_new classification score: {}".format(classify_score_model_new))
            _, indices_model_new = torch.topk(classify_score_model_new, 1)
            # print("model_new classification: {}".format(indices_model_new.t()))

            for alpha in temp_alpha_dict.keys():
                # print("alpha: {}".format(alpha))
                sum_of_score = classify_score_model_old + alpha * classify_score_model_new
                # print("model_combine classification score: {}".format(sum_of_score))
                _, indices_sum_of_score = torch.topk(sum_of_score, 1)
                # print("model_combine classification: {}".format(indices_sum_of_score.t()))

                # Compute the distance between feature and decision boundary
                dist_delta = sum(alpha * abs(classify_score_model_new / classify_score_model_old)) / \
                             classify_score_model_old.shape[0]
                # print("average distance change: {}".format(dist_delta.cpu().item()))
                temp_alpha_dict[alpha].append(dist_delta.cpu().item())

            temp_feature_mean_torch.cpu()

    # print("\n")
    return temp_alpha_dict


def get_each_previous_task_alpha_dict(temp_model,
                                      temp_model_old,
                                      task_ind,
                                      all_task_feature_label_dict,
                                      num_of_task_dict):

    each_previous_task_alpha_dict = dict()

    for task_i in range(0, task_ind):
        print("previous task: {}".format(task_i))
        temp_feature_label_dict = all_task_feature_label_dict[task_i]

        temp_alpha_dict_ = dict()
        for i in np.arange(2.0, 0.0, -0.05):
            temp_alpha_dict_[i] = list()
        temp_alpha_dict_[0] = list()

        each_previous_task_alpha_dict[task_i] = calculate_feature_decision_boundary_distance(temp_feature_label_dict,
                                                                                             temp_model, temp_model_old,
                                                                                             temp_alpha_dict_,
                                                                                             num_of_task_dict[task_i])

    return each_previous_task_alpha_dict


def calculate_num_of_cls(previous_task_index, temp_feature_label_dict, num_of_task_dict, class_number):
    num_of_cls = list()

    if previous_task_index == 0:
        num_of_cls = [k for k in range(0, class_number)]
    else:
        for cls in range(0, class_number):
            if temp_feature_label_dict[cls] == []:
                print("Task {} do not learn new subclass!".format(cls))
            else:
                num_of_cls.append(cls)

    num_of_task_dict[previous_task_index] = num_of_cls

    return num_of_task_dict


def calculate_last_step_feature(previous_task_index,
                                temp_model_old,
                                target_train_val_augment_task_loader_dict,
                                all_task_feature_label_dict,
                                num_class,
                                source_train_val_augment_dataset_loader
                                ):

    temp_feature_label_dict = dict()
    for i in range(0, num_class):
        temp_feature_label_dict[i] = list()

    temp_model_old.eval()

    with torch.no_grad():
        if previous_task_index == 0:
            print(" Use source_train_val_augment_dataset_loader for previous Task 0")
            target_train_val_augment_loader_temp = source_train_val_augment_dataset_loader
        else:
            print(" Use target_train_val_augment_task_loader_dict for previous Task {}".format(previous_task_index))
            target_train_val_augment_loader_temp = target_train_val_augment_task_loader_dict[previous_task_index]

        for j, (images, target) in enumerate(target_train_val_augment_loader_temp):
            images = images.cuda(non_blocking=True)
            _, feature = temp_model_old(images)
            feature = feature.cpu().numpy()
            if j == 0:
                print("feature.shape: {}".format(feature.shape))
            target = target.numpy()
            if j == 0:
                print("target.shape: {}".format(target.shape))
            for i in range(0, len(target)):
                temp_feature_label_dict[target[i]].append(feature[i])
            if j % 100 == 0:
                print(j)

    all_task_feature_label_dict[previous_task_index] = temp_feature_label_dict

    return all_task_feature_label_dict, temp_feature_label_dict


def calculate_val_top1_list(performance_dict_task):
    train_val_top1_name = 'train_val_top1'
    train_val_top1_list = list()

    for key in performance_dict_task.keys():
        train_val_top1_list.append(performance_dict_task[key][train_val_top1_name])

    return train_val_top1_list


def inc_trainer(model,
                model_old,
                task_index,
                target_train_train_task_loader_dict,
                target_val_val_task_loader_dict,
                target_train_val_task_loader_dict,
                class_number,
                IL_initial_lr,
                retrain_epoch,
                criterion,
                incremental_learning_momentum,
                incremental_learning_wd,
                logger):

    print("=> Task: {}, Initial LR: {}, Inc_Epoch: {}".format(task_index, IL_initial_lr, retrain_epoch))
    previous_task_list = sorted([i for i in range(1, task_index)], reverse=True)
    model.train

    # fixed updating for BN statistics.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False
            m.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    incremental_learning_momentum = incremental_learning_momentum
    incremental_learning_wd = incremental_learning_wd
    params = list()
    params.append({"params": filter(lambda p: p.requires_grad, model.module.last_linear.parameters()),
                   'weight_decay': incremental_learning_wd})
    optimizer = optim.SGD(params, IL_initial_lr, momentum=incremental_learning_momentum,
                          weight_decay=incremental_learning_wd)

    train_loader = target_train_train_task_loader_dict[task_index]
    val_loader = target_val_val_task_loader_dict[task_index]
    train_val_loader = target_train_val_task_loader_dict[task_index]
    best_acc1 = 0  # use to save the best top-1 model during training

    for epoch in range(0, retrain_epoch):
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # normal aug input
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with torch.no_grad():
                output_old, _ = model_old(input)
                output_old = F.softmax(output_old, dim=1)

            optimizer.zero_grad()
            output, _ = model(input)
            output = F.softmax(output, dim=1)

            num_class = class_number
            batch_size = input.size(0)
            z_c = output_old[torch.arange(batch_size), target]  # y^{c_{i}} * f^{t}(x_{i})
            z_c = torch.reshape(z_c, (batch_size, 1))
            z_c = z_c.repeat(1, num_class)  # y^{c_{i}} * f^{t}(x_{i})
            cita_pi = - torch.exp(- 0.5 * (z_c - output_old))  # # -exp(-1/2 * (y^{c_{i}} - y^{k}) * f^{t}(x_{i})), here y^{k} * f^{t}(x_{i}) is refer to output_old
            w_i = - (torch.sum(cita_pi, 1) - (-1.0))  # for each image, we have a specific w_{i}, the "-1" appear when k = c_{i}, which mean we have +(-1), then we need to -(-1)
            w_i = torch.reshape(w_i, (batch_size, 1))
            w_i_repeat = w_i.repeat(1, num_class)
            tao_k = (-1) * torch.div(cita_pi, w_i_repeat)

            g_c = output[torch.arange(batch_size), target]  # y^{c_{i}} * g(x_{i})
            g_c = torch.reshape(g_c, (batch_size, 1))
            g_c = g_c.repeat(1, num_class)  # y^{c_{i}} * g(x_{i})

            #  functional gradient of the empirical risk with normalization
            emp_minus_delta_R = -torch.sum(w_i * torch.reshape(torch.sum((g_c - output) * tao_k, 1), (batch_size, 1))) / (batch_size)

            print("emp_minus_delta_R: {}".format(emp_minus_delta_R))

            emp_minus_delta_R.backward()
            optimizer.step()
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                     batch_time=batch_time,data_time=data_time,
                                                                     loss=losses, top1=top1, top5=top5))

        acc1, _, _ = validate(val_loader, model, criterion,logger)
        train_val_acc1, _, _ = validate(train_val_loader, model, criterion,logger)
        # remember best acc@1 and save checkpoint
        is_best = train_val_acc1 > best_acc1
        best_acc1 = max(train_val_acc1, best_acc1)
        if is_best:
            best_model_state = deepcopy(model.state_dict())
            print("best_val_acc1: {}".format(best_acc1))

        print("\n")

    return model, best_model_state, best_acc1


if __name__ == '__main__':
    main()
