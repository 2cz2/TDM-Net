import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        vocab_size = input.size(-1)
        device = input.device

        weights = torch.ones(vocab_size, device=device)
        weights[4] = 0.1
        weights[3] = 0.5
        weights[0] = 0.0

        loss_fn = nn.CrossEntropyLoss(weight=weights, reduction='none')

        input = input.contiguous().view(-1, vocab_size)
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        loss = loss_fn(input, target)

        total_valid = torch.max(mask.sum(), torch.tensor(1.0, device=device))
        output = torch.sum(loss * mask) / total_valid

        return output


def compute_loss(output, reports, masks):
    criterion = LanguageModelCriterion()
    return criterion(output, reports, masks)