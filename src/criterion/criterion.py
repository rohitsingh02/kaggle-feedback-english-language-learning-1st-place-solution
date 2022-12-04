import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class MCRMSELoss(nn.Module):
    def __init__(self, num_scored=6, weights=None):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored
        self.weights = [1/num_scored for _ in range(num_scored)] if weights is None else weights

    def forward(self, yhat, y):
        score = 0
        for i, w in enumerate(self.weights):
            score += self.rmse(yhat[:, :, i], y[:, :, i]) * w
        return score


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor.argmax(dim=1),
            weight=self.weight,
            reduction=self.reduction,
        )


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class WeightedDenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)

        if weights is not None:
            loss = loss * weights
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.mean()

        return loss


def get_criterion(config):

    if config.criterion.criterion_type == 'SmoothL1Loss':
        return torch.nn.SmoothL1Loss(
            reduction=config.criterion.smooth_l1_loss.reduction,
            beta=config.criterion.smooth_l1_loss.beta
        )

    elif config.criterion.criterion_type == 'RMSELoss':
        return RMSELoss(
            eps=config.criterion.rmse_loss.eps,
            reduction=config.criterion.rmse_loss.reduction
        )

    elif config.criterion.criterion_type == 'MCRMSELoss':
        return MCRMSELoss(
            weights=config.criterion.mcrmse_loss.weights,
        )

    return nn.MSELoss()
