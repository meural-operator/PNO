import torch
import json
import os

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=1e-5):
        super(UnitGaussianNormalizer, self).__init__()
        # x could be in shape of [N, C, M]
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps
            mean = self.mean
        else:
            std = self.std[sample_idx] + self.eps
            mean = self.mean[sample_idx]
        x = (x * std) + mean
        return x

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

import os

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms
    
    def __call__(self, x, y):
        return self.rel(x, y)

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # Clear log if exists
        open(log_path, 'w').close()
        
    def log(self, step, metrics):
        record = {'step': step}
        record.update(metrics)
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
