import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point


def calcScaleZeroPointSym(min_val, max_val, num_bits=8):
    # Calc Scale
    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.

    scale = max_val / qmax

    return scale, 0


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


def quantize_tensor_sym(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.

    scale = max_val / qmax

    q_x = x / scale

    q_x.clamp_(-qmax, qmax).round_()
    q_x = q_x.round()
    return QTensor(tensor=q_x, scale=scale, zero_point=0)


def dequantize_tensor_sym(q_x):
    return q_x.scale * (q_x.tensor.float())


# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key):
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)

    # add ema calculation

    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += 1

    weighting = 2.0 / (stats[key]['total']) + 1

    if 'ema_min' in stats[key]:
        stats[key]['ema_min'] = weighting * (min_val.mean().item()) + (1 - weighting) * stats[key]['ema_min']
    else:
        stats[key]['ema_min'] = weighting * (min_val.mean().item())

    if 'ema_max' in stats[key]:
        stats[key]['ema_max'] = weighting * (max_val.mean().item()) + (1 - weighting) * stats[key]['ema_max']
    else:
        stats[key]['ema_max'] = weighting * (max_val.mean().item())

    stats[key]['min_val'] = stats[key]['min'] / stats[key]['total']
    stats[key]['max_val'] = stats[key]['max'] / stats[key]['total']

    return stats


class QuadbitMnistModel(nn.Module):
    def __init__(self, n_pixel):
        super(QuadbitMnistModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               padding_mode="replicate")
        self.batchnorm1 = nn.BatchNorm2d(num_features=32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        n_pixel = n_pixel // 2

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               padding_mode="replicate")
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        n_pixel = n_pixel // 2

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               padding_mode="replicate")
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=64 * (n_pixel ** 2), out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)
        self.dropout = nn.Dropout(p=.5)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = self.activation(x)

        return x


class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8, min_val=None, max_val=None):
        x = quantize_tensor(x,num_bits=num_bits, min_val=min_val, max_val=max_val)
        x = dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None


class QuantAwareTraining:
    def __init__(self, model, num_bits=4, act_quant=False):
        n_pixel = 28
        self.model = model
        self.num_bits = num_bits
        self.act_quant = act_quant

    def quantAwareTrainingForward(self, x, stats, vis=False, axs=None, sym=False, act_quant=False):
        conv1weight = self.model.conv1.weight.data
        self.model.conv1.weight.data = FakeQuantOp.apply(self.model.conv1.weight.data, self.num_bits)
        x = self.model.conv1(x)
        x = F.relu(x)

        with torch.no_grad():
            stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')
        if act_quant:
            x = FakeQuantOp.apply(x, self.num_bits, stats['conv1']['ema_min'], stats['conv1']['ema_max'])

        x = F.max_pool2d(x, 2, 2)

        conv2weight = self.model.conv2.weight.data
        self.model.conv2.weight.data = FakeQuantOp.apply(self.model.conv2.weight.data, self.num_bits)
        x = self.model.conv2(x)
        x = F.relu(x)

        with torch.no_grad():
            stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
        if act_quant:
            x = FakeQuantOp.apply(x, self.num_bits, stats['conv2']['ema_min'], stats['conv2']['ema_max'])

        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)

        x = self.model.fc1(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.model.fc2(x)
        x = self.model.activation(x)

        return x, conv1weight, conv2weight, stats
