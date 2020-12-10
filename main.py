from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import train
from models.resnet import *
from utils import *

from dataloader import CIFAR10, CIFAR100
import noise_model

parser = argparse.ArgumentParser(description='Noise Label Learning')

parser.add_argument('--method', default='trunc_loss', type=str)

#shared parameter
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--batch-size', '-b', default=128,type=int, help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=220, type=int, help='number of total epochs to run')
parser.add_argument('--lr', default=0.01, type=float,help='initial learning rate')
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--sess', default='default', type=str, help='session id')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', dest='gpu', default='0', type=str,help='select gpu')
parser.add_argument('--dir', dest='dir', default='/tmp/pycharm_project_680/result_pencl', type=str, metavar='PATH',help='model dir')
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--schedule', nargs='+', type=int)

#trunc_loss
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
parser.add_argument('--start_prune', default=40, type=int,help='number of total epochs to run')
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--schedule', nargs='+', type=int)

#PENCL
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--lr2', '--learning-rate2', default=0.2, type=float, metavar='H-P', help='initial learning rate of stage3')
parser.add_argument('--alpha', default=0.4, type=float,metavar='H-P', help='the coefficient of Compatibility Loss')
parser.add_argument('--beta', default=0.1, type=float,metavar='H-P', help='the coefficient of Entropy Loss')
parser.add_argument('--lambda1', default=600, type=int,metavar='H-P', help='the value of lambda')
parser.add_argument('--stage1', default=70, type=int, metavar='H-P', help='number of epochs utill stage1')
parser.add_argument('--stage2', default=200, type=int,metavar='H-P', help='number of epochs utill stage2')
parser.add_argument('--datanum', default=50000, type=int, metavar='H-P', help='number of train dataset samples')
parser.add_argument('--classnum', default=10, type=int,metavar='H-P', help='number of train dataset classes')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
parser.add_argument('--pretrained', default=False, dest='pretrained', action='store_true',help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,help='distributed backend')


#MLNT
parser.add_argument('--meta_lr', default=0.02, type=float, help='meta learning_rate')
parser.add_argument('--num_fast', default=10, type=int, help='number of random perturbations')
parser.add_argument('--perturb_ratio', default=0.5, type=float, help='ratio of random perturbations')
parser.add_argument('--start_iter', default=500, type=int)
parser.add_argument('--mid_iter', default=2000, type=int)
parser.add_argument('--start_epoch', default=80, type=int)
parser.add_argument('--alpha', default=1, type=int)
parser.add_argument('--eps', default=0.99, type=float, help='Running average of model weights')

#co_teaching
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--fr_type', type = str, help='forget rate type', default='type_1')

args = parser.parse_args()


def main():
    use_cuda = torch.cuda.is_available()
    global best_acc

    # load dataset
    if args.dataset == 'cifar10':

        num_classes = 10

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ])

        train_dataset = CIFAR10(root='./data/',
                                download=True,
                                train=True,
                                transform=transform_train,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

        test_dataset = CIFAR10(root='./data/',
                               download=True,
                               train=False,
                               transform=transform_test,
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                               )

    if args.dataset == 'cifar100':

        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
        ])
        train_dataset = CIFAR100(root='./data/',
                                 download=True,
                                 train=True,
                                 transform=transform_train,
                                 noise_type=args.noise_type,
                                 noise_rate=args.noise_rate
                                 )

        test_dataset = CIFAR100(root='./data/',
                                download=True,
                                train=False,
                                transform=transform_test,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

    elif args.dataset == 'Clothing1M':
        loader = dataloader.clothing_dataloader(batch_size=args.batch_size, shuffle=True)
        train_dataset, val_dataset, test_dataset = loader.run()

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    print('==> Building model.. (Default : ResNet34)')
    start_epoch = 0
    net = ResNet34(num_classes)

    result_folder = './results/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    logname = result_folder + net.__class__.__name__ + args.model + args.noise_type + str(args.noise_rate)+\
              '_' + args.sess + '.csv'

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        print('Using CUDA..')

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train loss', 'train acc','val loss', 'val acc', 'test loss', 'test acc'])


    if args.method == 'trunc_loss':
        criterion = noise_model.TruncatedLoss(trainset_size=len(train_dataset)).cuda()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.schedule, gamma=args.gamma)

    elif args.method == 'PENCL':

        y_file = "./y.npy"
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    elif args.method == 'MLNT':
        criterion = nn.CrossEntropyLoss()
        consistent_criterion = nn.KLDivLoss()
        tch_net = ResNet34(num_classes).cuda()
        pretrain_net = ResNet34(num_classes, pretrained=True).cuda()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    elif args.method == 'co-teaching':
        optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)
        clf2 = ResNet34(num_classes)
        optimizer2 = torch.optim.Adam(clf2.parameters(), lr=learning_rate)

    for epoch in range(start_epoch, args.epochs):

        if args.method == 'trunc_loss':
            train.trunc_train(epoch, trainloader, net, criterion, optimizer)
            test_loss, test_acc = train.trunc_test(epoch, testloader, net, criterion,best_acc)

        elif args.method == 'PENCL':
            if os.path.isfile(y_file):
                y = np.load(y_file)
            else:
                y = []
            noise_model.PENCL.adjust_learning_rate(optimizer, epoch)
            train.pencl_train(train_loader, model, criterion, optimizer, epoch, y)
            test_loss, test_acc = train.pencl_validate(testloader, model, criterion, best_acc)

        elif args.method == 'MLNT':
            train.mlnt_train(net, tch_net, optimizer, criterion, epoch, train_loader, pretrain_net, consistent_criterion)
            train.mlnt_val(net, test_loader, criterion)
            test_loss, test_acc = train.mlnt_val_tch(tch_net, test_loader, criterion)
            net.train()
            tch_net.train()
            scheduler.step()

        elif args.method == 'co-teaching':
            net.train()
            clf2.train()
            adjust_learning_rate(optimizer1, epoch)
            adjust_learning_rate(optimizer2, epoch)
            train_acc1, train_acc2 = train.co_teaching_train(train_loader, epoch, clf1, optimizer1, clf2, optimizer2)
            test_acc1, test_acc2 = train.co_teaching_evaluate(test_loader, clf1, clf2)

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
        scheduler.step()

    print(f'best ass {best_acc}')


# Training



def checkpoint(acc, epoch, net, best):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'best_acc': best
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + args.model + args.noise_type+ str(args.noise_rate) + args.sess)


if __name__ == '__main__':
    main()

