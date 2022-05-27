import torch.nn as nn
import torch
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=3, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
focal_loss=FocalLoss()
class WeightLoss(nn.Module):
    def __init__(self):
        super(WeightLoss, self).__init__()


    def forward(self, inputs, targets):

        BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        weight=torch.sigmoid(torch.abs(inputs-targets))
        a=torch.ones_like(weight)
        W=torch.pow((weight-a*0.5),2)
        W_loss=BCE_loss.mul(W)

        return torch.mean(W_loss)
focal_loss=FocalLoss()
WeightLoss=WeightLoss()