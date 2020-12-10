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

from models.resnet import *
from utils import *

from data.cifar import CIFAR10, CIFAR100
import noise_model


def trunc_train(epoch, trainloader, net, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if (epoch + 1) >= args.start_prune and (epoch + 1) % 10 == 0:
        checkpoint = torch.load('./checkpoint/' + args.model + args.noise_type+ str(args.noise_rate) + args.sess)
        net = checkpoint['net']
        net.eval()
        for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            criterion.update_weight(outputs, targets, indexes)
        now = torch.load('./checkpoint/current_net')
        net = now['current_net']
        net.train()

    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets, indexes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return (train_loss / batch_idx, 100. * correct / total)


def trunc_test(epoch, testloader, net, criterion,best_acc):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets, indexes)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch, net, best_acc)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    state = {
        'current_net': net,
    }
    torch.save(state, './checkpoint/current_net')
    return (test_loss / batch_idx, 100. * correct / total)


def pencl_train(train_loader, model, criterion, optimizer, epoch, y):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    # new y is y_tilde after updating
    new_y = np.zeros([args.datanum, args.classnum])

    for i, (input, target, index,_) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)

        index = index.numpy()

        target1 = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target1)

        # compute output
        output = model(input_var.cuda())

        logsoftmax = nn.LogSoftmax(dim=1).cuda()
        softmax = nn.Softmax(dim=1).cuda()
        if epoch < args.stage1:
            # lc is classification loss
            lc = criterion(output, target_var)
            # init y_tilde, let softmax(y_tilde) is noisy labels
            onehot = torch.zeros(target.size(0), 10).scatter_(1, target.view(-1, 1), 10.0)
            onehot = onehot.numpy()
            new_y[index, :] = onehot
        else:
            yy = y
            yy = yy[index, :]
            yy = torch.FloatTensor(yy)
            yy = yy.cuda()
            yy = torch.autograd.Variable(yy, requires_grad=True)
            # obtain label distributions (y_hat)
            last_y_var = softmax(yy)
            lc = torch.mean(softmax(output) * (logsoftmax(output) - torch.log((last_y_var))))
            # lo is compatibility loss
            lo = criterion(last_y_var, target_var)
        # le is entropy loss
        le = - torch.mean(torch.mul(softmax(output), logsoftmax(output)))

        if epoch < args.stage1:
            loss = lc
        elif epoch < args.stage2:
            loss = lc + args.alpha * lo + args.beta * le
        else:
            loss = lc

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target1, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >= args.stage1 and epoch < args.stage2:
            lambda1 = args.lambda1
            # update y_tilde by back-propagation
            yy.data.sub_(lambda1 * yy.grad.data)

            new_y[index, :] = yy.data.cpu().numpy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if epoch < args.stage2:
        # save y_tilde
        y = new_y
        y_file = args.dir + "y.npy"
        np.save(y_file, y)
        y_record = args.dir + "record/y_%03d.npy" % epoch
        np.save(y_record, y)


def pencl_validate(val_loader, model, criterion,best_acc):
    batch_time = noise_model.PENCL()
    losses = noise_model.PENCL()
    top1 = noise_model.PENCL()
    top5 = noise_model.PENCL()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target,_,_) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var.cuda())
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if top1.avg > best_acc:
        best_acc = top1.avg
        checkpoint(top1.avg, epoch, net, best_acc)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    return losses.avg(), top1.avg


def mlnt_train(net,tch_net,optimizer,criterion,epoch,train_loader,pretrain_net,consistent_criterion,args):
    global init
    net.train()
    tch_net.train()
    train_loss = 0
    correct = 0
    total = 0

    learning_rate = args.lr
    if epoch > args.start_epoch:
        learning_rate = learning_rate / 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    #print('\n=> %s Training Epoch #%d, LR=%.4f' % (args.id, epoch, learning_rate))

    for batch_idx, (inputs, targets,_,_) in enumerate(train_loader):

        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)  # Forward Propagation

        class_loss = criterion(outputs, targets)  # Loss
        class_loss.backward(retain_graph=True)

        if batch_idx > args.start_iter or epoch > 1:
            if batch_idx > args.mid_iter or epoch > 1:
                args.eps = 0.999
                alpha = args.alpha
            else:
                u = (batch_idx - args.start_iter) / (args.mid_iter - args.start_iter)
                alpha = args.alpha * math.exp(-5 * (1 - u) ** 2)

            if init:
                init = False
                for param, param_tch in zip(net.parameters(), tch_net.parameters()):
                    param_tch.data.copy_(param.data)
            else:
                for param, param_tch in zip(net.parameters(), tch_net.parameters()):
                    param_tch.data.mul_(args.eps).add_((1 - args.eps), param.data)

            _, feats = pretrain_net(inputs, get_feat=True)
            tch_outputs = tch_net(inputs, get_feat=False)
            p_tch = F.softmax(tch_outputs, dim=1)
            p_tch.detach_()

            for i in range(args.num_fast):
                targets_fast = targets.clone()
                randidx = torch.randperm(targets.size(0))
                for n in range(int(targets.size(0) * args.perturb_ratio)):
                    num_neighbor = 10
                    idx = randidx[n]
                    feat = feats[idx]
                    feat.view(1, feat.size(0))
                    feat.data = feat.data.expand(targets.size(0), feat.size(0))
                    dist = torch.sum((feat - feats) ** 2, dim=1)
                    _, neighbor = torch.topk(dist.data, num_neighbor + 1, largest=False)
                    targets_fast[idx] = targets[neighbor[random.randint(1, num_neighbor)]]

                fast_loss = criterion(outputs, targets_fast)

                grads = torch.autograd.grad(fast_loss, net.parameters(), create_graph=True, retain_graph=True,
                                            only_inputs=True)
                for grad in grads:
                    grad.detach()
                    #grad.requires_grad = False

                fast_weights = OrderedDict(
                    (name, param - args.meta_lr * grad) for ((name, param), grad) in zip(net.named_parameters(), grads))

                fast_out = net.forward(inputs, fast_weights)

                logp_fast = F.log_softmax(fast_out, dim=1)
                consistent_loss = consistent_criterion(logp_fast, p_tch)
                consistent_loss = consistent_loss * alpha / args.num_fast
                consistent_loss.backward()

        optimizer.step()  # Optimizer update

        train_loss += class_loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, args.epochs, batch_idx + 1, (len(train_loader.dataset) // args.batch_size) + 1,
                            class_loss.item(), 100. * correct / total))
        sys.stdout.flush()



def mlnt_val(net,test_loader,criterion, best):
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets,_,_) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100. * correct / total

    if acc > best:
        best = acc
        print('| Saving Best Model (net)...')
        checkpoint(acc, epoch, net, best)


def mlnt_val_tch(tch_net,test_loader,criterion,best):
    tch_net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets,_,_) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = tch_net(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100. * correct / total

    if acc > best:
        best = acc
        print('| Saving Best Model (tchnet)...')
        save_point = './checkpoint/%s.pth.tar' % (args.id)
        checkpoint(acc, epoch, net, best)

