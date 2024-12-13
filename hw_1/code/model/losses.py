import torch
from torch.nn import functional as F  # noqa: N812, WPS347

class CrossEntropyLossWeighted(torch.nn.Module):
    def __init__(
        self,
        reduction='mean',
        weight=None,
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = torch.tensor(weight, dtype=torch.float, device='cuda') if weight is not None else weight

    def forward(self, logits, labels):
        loss = F.cross_entropy(
            logits,
            labels.to(torch.long) if labels.dtype == torch.double else labels,
            reduction=self.reduction,
            weight=self.weight,
        )
        return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = F.cross_entropy(
            input_,
            target.to(torch.long) if target.dtype == torch.double else target,
            reduction='none',
        )
        if len(target.shape) > 1:  # in case of using mixer or probs
            _, target = torch.max(target, 1)
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.to(torch.long).unsqueeze(1)).flatten()
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss)
