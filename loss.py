import torch
import torch.nn as nn
from torch.nn.functional import nll_loss, cross_entropy
from torch.autograd import Variable
import torch.nn.functional as F 

class BinaryCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.bce_loss(inputs, targets)

# DICE = 2 * Sum(PiGi) / (Sum(Pi) + Sum(Gi))
# Refer https://github.com/pytorch/pytorch/issues/1249 for Laplace/Additive smooth
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets):
        smooth = 1.
        num = targets.size(0) # number of batches
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        dice = score.sum() / num
        # three kinds of loss formulas: (1) 1 - dice (2) -dice (3) -torch.log(dice)
        return 1. - dice


class Softmax_SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets):
        smooth = 1.
        num = targets.size(0) # number of batches
        num_class = inputs.size(1)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1, num_class):
            m1 = inputs[:,i,:,:].view(num, -1)
            m2 = (targets==i).float().view(num, -1)
            intersection = (m1 * m2)
            score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            Dice += score.sum()
        dice = Dice / num
        # print('dice loss', dice)
        # three kinds of loss formulas: (1) 1 - dice (2) -dice (3) -torch.log(dice)
        return 1. - dice



# Jaccard/IoU = Sum(PiGi) / (Sum(Pi) + Sum(Gi) - Sum(PiGi))
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets):
        smooth = 1.
        num = targets.size(0) # number of batches
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + smooth)
        iou = score.sum() / num
        # three kinds of loss formulas: (1) 1 - iou (2) -iou (3) -torch.log(iou)
        return 1. - iou

class Softmax_IoULoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, weights=None):
        smooth = 1.
        num = targets.size(0) # number of batches
        num_class = inputs.size(1)
        ious = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1, num_class):
            m1 = inputs[:,i,:,:].view(num, -1)
            m2 = (targets==i).float().view(num, -1)
            intersection = (m1 * m2)
            score =  (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + smooth)
            # ious += (num -score.sum()) * weights[0,i,0,0]
            ious += score.sum()
        iou = ious / num
        # print('iou:', iou)
        # print(abc)
        # three kinds of loss formulas: (1) 1 - iou (2) -iou (3) -torch.log(iou)
        return 1- iou 



class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection [https://arxiv.org/abs/1708.02002]
    Digest the paper as below:
        α, balances the importance of positive/negative examples
        γ, focusing parameter that controls the strength of the modulating term

            CE(pt) = −log(pt) ==> pt = exp(-CE)
            FL(pt) = −α((1 − pt)^γ) * log(pt)

        In general α should be decreased slightly as γ is increased (for γ = 2, α = 0.25 works best).
    """
    def __init__(self, focusing_param=2, balance_param=0.25, size_average=True):
        super().__init__()
        self.gamma = focusing_param
        self.alpha = balance_param
        self.size_average = size_average

    def forward(self, inputs, targets, weights=None):
        logpt = -F.binary_cross_entropy(inputs, targets, weights, size_average=self.size_average)
        pt = torch.exp(logpt)
        # compute the loss
        focal_loss = -((1-pt)**self.gamma) * logpt
        balanced_focal_loss = self.alpha * focal_loss
        return balanced_focal_loss


class Softmax_FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection [https://arxiv.org/abs/1708.02002]
    Digest the paper as below:
        α, balances the importance of positive/negative examples
        γ, focusing parameter that controls the strength of the modulating term

            CE(pt) = −log(pt) ==> pt = exp(-CE)
            FL(pt) = −α((1 − pt)^γ) * log(pt)

        In general α should be decreased slightly as γ is increased (for γ = 2, α = 0.25 works best).
    """
    def __init__(self, focusing_param=2, balance_param=0.25, size_average=True):
        super().__init__()
        self.gamma = focusing_param
        self.alpha = balance_param
        self.size_average = size_average

    def forward(self, input, target, weights=None):
        logpt = -F.cross_entropy(input, target, weights, size_average=self.size_average)
        pt = torch.exp(logpt)
        # compute the loss
        focal_loss = -((1-pt)**self.gamma) * logpt
        balanced_focal_loss = self.alpha * focal_loss
        return balanced_focal_loss



def criterion(preds, labels):
# (1) BCE Loss
#     return BinaryCrossEntropyLoss2d().forward(preds, labels)
# (2) BCE Loss + DICE Loss
#     return BinaryCrossEntropyLoss2d().forward(preds, labels) + \
#            SoftDiceLoss().forward(preds, labels)
# (3) BCE Loss + Jaccard/IoU Loss
    return BinaryCrossEntropyLoss2d().forward(preds, labels) + \
           IoULoss().forward(preds, labels)

def segment_criterion(preds, labels):
    return BinaryCrossEntropyLoss2d().forward(preds, labels) + \
           IoULoss().forward(preds, labels)

def contour_criterion(preds, labels):
    return IoULoss().forward(preds, labels)

def weight_criterion(preds, labels, weights):
    return binary_cross_entropy(preds, labels, weights) + \
           IoULoss().forward(preds, labels)

def focal_criterion(preds, labels, weights):
    return FocalLoss().forward(preds, labels, weights) + \
           SoftDiceLoss().forward(preds, labels)
            # IoULoss().forward(preds, labels)

def softmax_focal_criterion(preds, labels):
    labels_classify = labels.long()
    return Softmax_FocalLoss().forward(preds, labels_classify)