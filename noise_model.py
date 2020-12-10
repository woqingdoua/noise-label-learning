import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)

    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1 - (Yg ** self.q)) / self.q) * self.weight[indexes] - ((1 - (self.k ** self.q)) / self.q) * \
               self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1 - (Yg ** self.q)) / self.q)
        Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)


class PENCL(object):

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

    def pencl_adjust_learning_rate(self,optimizer, epoch):
        """Sets the learning rate"""
        if epoch < args.stage2:
            lr = args.lr
        elif epoch < (args.epochs - args.stage2) // 3 + args.stage2:
            lr = args.lr2
        elif epoch < 2 * (args.epochs - args.stage2) // 3 + args.stage2:
            lr = args.lr2 // 10
        else:
            lr = args.lr2 // 100
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



def Co_teaching():

    def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
        loss_1 = F.cross_entropy(y_1, t, reduction='none')
        ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, t, reduction='none')
        ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember].cpu()
        ind_2_update = ind_2_sorted[:num_remember].cpu()
        if len(ind_1_update) == 0:
            ind_1_update = ind_1_sorted.cpu().numpy()
            ind_2_update = ind_2_sorted.cpu().numpy()
            num_remember = ind_1_update.shape[0]

        pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_update]]) / float(num_remember)
        pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_update]]) / float(num_remember)

        loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

        return torch.sum(loss_1_update) / num_remember, torch.sum(
            loss_2_update) / num_remember, pure_ratio_1, pure_ratio_2

    def loss_coteaching_plus(logits, logits2, labels, forget_rate, ind, noise_or_not, step):
        outputs = F.softmax(logits, dim=1)
        outputs2 = F.softmax(logits2, dim=1)

        _, pred1 = torch.max(logits.data, 1)
        _, pred2 = torch.max(logits2.data, 1)

        pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

        logical_disagree_id = np.zeros(labels.size(), dtype=bool)
        disagree_id = []
        for idx, p1 in enumerate(pred1):
            if p1 != pred2[idx]:
                disagree_id.append(idx)
                logical_disagree_id[idx] = True

        temp_disagree = ind * logical_disagree_id.astype(np.int64)
        ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
        try:
            assert ind_disagree.shape[0] == len(disagree_id)
        except:
            disagree_id = disagree_id[:ind_disagree.shape[0]]

        _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
        update_step = Variable(torch.from_numpy(_update_step)).cuda()

        if len(disagree_id) > 0:
            update_labels = labels[disagree_id]
            update_outputs = outputs[disagree_id]
            update_outputs2 = outputs2[disagree_id]

            loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(update_outputs, update_outputs2, update_labels,
                                                                         forget_rate, ind_disagree, noise_or_not)
        else:
            update_labels = labels
            update_outputs = outputs
            update_outputs2 = outputs2

            cross_entropy_1 = F.cross_entropy(update_outputs, update_labels)
            cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels)

            loss_1 = torch.sum(update_step * cross_entropy_1) / labels.size()[0]
            loss_2 = torch.sum(update_step * cross_entropy_2) / labels.size()[0]

            pure_ratio_1 = np.sum(noise_or_not[ind]) / ind.shape[0]
            pure_ratio_2 = np.sum(noise_or_not[ind]) / ind.shape[0]
        return loss_1, loss_2, pure_ratio_1, pure_ratio_2



