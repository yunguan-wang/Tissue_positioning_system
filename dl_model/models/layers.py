import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import numbers

def get_norm_layer_and_bias(norm_layer='batch', use_bias=None):
    """ Return a normalization layer and set up use_bias for convoluation layers.
    
    Parameters:
        norm_layer: (str) -- the name of the normalization layer: [batch, instance]
                    None -- no batch norm
                    other module: nn.BatchNorm2d, nn.InstanceNorm2d

    For BatchNorm: use learnable affine parameters. (affine=True)
                   track running statistics (mean/stddev). (track_running_stats=True)
                   do not use bias in previous convolution layer. (use_bias=False)
    For InstanceNorm: do not use learnable affine parameters. (affine=False)
                      do not track running statistics. (track_running_stats=False)
                      use bias in previous convolution layer. (use_bias=True)
    Test commands:
        get_norm_layer_and_bias('batch', None) -> affine=True, track_running_stats=True, False
        get_norm_layer_and_bias('batch', True) -> affine=True, track_running_stats=True, True
        get_norm_layer_and_bias('instance', None) -> affine=False, track_running_stats=False, True
        get_norm_layer_and_bias('instance', False) -> affine=False, track_running_stats=False, False
        get_norm_layer_and_bias(None, None) -> None, True
        get_norm_layer_and_bias(None, False) -> None, False
        get_norm_layer_and_bias(nn.BatchNorm2d, None) -> BatchNorm2d, False
        get_norm_layer_and_bias(nn.BatchNorm2d, True) -> BatchNorm2d, True
        get_norm_layer_and_bias(nn.InstanceNorm2d, None) -> InstanceNorm2d, True
        get_norm_layer_and_bias(nn.InstanceNorm2d, False) -> InstanceNorm2d, False
    """
    if isinstance(norm_layer, str):
        if norm_layer == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('normalization layer {} is not found'.format(norm_layer))
    
    if use_bias is None:
        use_bias = norm_layer == nn.InstanceNorm2d
    
    return norm_layer, use_bias


class ConvBNReLU(nn.Sequential):
    def __init__(self, conv, norm_layer=None, activation=None, dropout_rate=0.0):
        ## get norm layer:
        if isinstance(norm_layer, str):
            norm_layer = get_norm_layer(norm_layer)
        
        layers = [conv]
        if norm_layer is not None:
            layers.append(norm_layer(conv.out_channels))
        if activation is not None:
            layers.append(activation)
        if dropout_rate:
            layers.append(nn.Dropout2d(dropout_rate))
        
        super(ConvBNReLU, self).__init__(*layers)


class Conv2dBNReLU(ConvBNReLU):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding='default', dilation=1, groups=1, bias=None, 
                 norm_layer='batch', activation=nn.ReLU(inplace=True), 
                 dropout_rate=0.0, 
                ):
        """ Create a Conv2d->BN->ReLU layer. 
            norm_layer: batch, instance, None
            activation: a nn layer.
            padding: 
                'default' (default): torch standard symmetric padding with (kernel_size - 1) // 2.
                int: symmetric padding to pass to nn.Conv2d(padding=padding)
                "same": tf padding="same", asymmetric for even kernel (l_0, r_1), etc)
                "valid": tf padding="valid", same as padding=0
        """
        ## get norm layer:
        norm_layer, bias = get_norm_layer_and_bias(norm_layer, bias)
        ## use Conv2d (extended nn.Conv2d) to support padding options
        if isinstance(kernel_size, numbers.Number):
            kernel_size = (kernel_size, kernel_size)
        if padding == 'default':
            padding = tuple((k-1)//2 for k in kernel_size)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                         padding, dilation, groups, bias=bias)
        super(Conv2dBNReLU, self).__init__(conv, norm_layer, activation, dropout_rate)


class DepthwiseConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding='default', dilation=1, groups=None, bias=None, expand_ratio=1, 
                 norm_layer='batch', activation=nn.ReLU(inplace=True)):
        """ Depthwise/Group convolution. 
            directly set bias based on norm_layer
        """
        inner_channels = int(expand_ratio * in_channels)
        groups = groups or int(np.gcd(in_channels, inner_channels))
        norm_layer, bias = get_norm_layer_and_bias(norm_layer, bias)
        
        super(DepthwiseConv2d, self).__init__(
            Conv2dBNReLU(in_channels, inner_channels, kernel_size, stride, 
                         padding, dilation, groups=groups, bias=bias, 
                         norm_layer=norm_layer, activation=activation),
            nn.Conv2d(inner_channels, out_channels, kernel_size=1, stride=1, 
                      padding=0, dilation=1, groups=1, bias=bias),
        )
        
        ## register values
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups


class InvertedResidual(nn.Module):
    """ https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding='default', dilation=1, groups=None, bias=None, expand_ratio=1,
                 norm_layer='batch', activation=nn.ReLU(inplace=True)):
        super(InvertedResidual, self).__init__()
        inner_channels = int(round(expand_ratio * in_channels))
        norm_layer, bias = get_norm_layer_and_bias(norm_layer, bias)
        assert stride in [1, 2]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups or inner_channels
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2dBNReLU(
                in_channels, inner_channels, kernel_size=1, # stride=1, padding='default' (0 for ks=1), 
                bias=bias, norm_layer=norm_layer, activation=activation))
        layers.extend([
            # dw
            Conv2dBNReLU(inner_channels, inner_channels, kernel_size=kernel_size, 
                         stride=stride, padding=padding, groups=self.groups, 
                         bias=bias, norm_layer=norm_layer, activation=activation),
            # pw-linear
            nn.Conv2d(inner_channels, out_channels, kernel_size=1, bias=bias), # stride=1, padding='default' (0 for ks=1), 
            norm_layer(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DepthwiseConv2dBNReLU6(ConvBNReLU):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, 
                 expand_ratio=1, padding='default', groups=None, norm_layer='batch', 
                 activation=nn.ReLU6(inplace=True), dropout_rate=0.0):
        """ Depthwise convolution BN Relu6. 
            (put kernel_size after stride to make it callable for 
             mobilenet_v2 constructor).
        """
        norm_layer, bias = get_norm_layer_and_bias(norm_layer)
        conv = DepthwiseConv2d(in_channels, out_channels, kernel_size,
                               stride, padding, groups=groups, bias=bias,
                               norm_layer=norm_layer, activation=activation)
        
        super(DepthwiseConv2dBNReLU6, self).__init__(conv, norm_layer, activation, dropout_rate)


################################################################
###################### Losses and Metrics ######################
################################################################
## not correct, F.kl_div: lambda input, target: target * (np.log(target)-input).
## What we want is log_softmax(input) * target
class SoftCrossEntropyLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(SoftCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # loss = F.kl_div(F.log_softmax(input), target, reduction='none')
        loss = F.log_softmax(input) * target
        if self.weight is not None:
            loss = loss * self.weight
        
        idx = [_ != self.ignore_index for _ in range(loss.shape[-1])]
        loss = loss[..., idx]
        loss = loss.sum(-1)
        
        if self.reduction == 'none':
            ret = loss
        elif self.reduction == 'mean':
            ret = loss.mean()
        elif self.reduction == 'sum':
            ret = loss.sum()
        else:
            ret = input
            raise ValueError(reduction + " is not valid")
        
        return ret


def match_pred_true(y_pred, y_true, binary=False, axis=1):
    """ Transform (sparse) y_true to match y_pred. """
    dtype = y_pred.dtype
    num_classes = y_pred.shape[axis]
    o = list(range(y_pred.ndim))
    o = o[0:axis] + [o[-1]] + o[axis:-1]
    
    ## squeeze if channel dimension is 1
    if y_true.ndim == y_pred.ndim and y_true.shape[axis] == 1:
        y_true = y_true.squeeze(axis)
    if y_true.ndim != y_pred.ndim:
        y_true = F.one_hot(y_true.type(torch.long), num_classes).permute(*o)
    
    if binary:
        y_true = F.one_hot(y_true.argmax(axis), num_classes).permute(*o)
        y_pred = F.one_hot(y_pred.argmax(axis), num_classes).permute(*o)
    
    return y_pred.to(dtype), y_true.to(dtype)


class IoU(nn.Module):
    def __init__(self, mode='iou', axis=1, eps=0.):
        """ Return a matrix of [batch * num_classes]. 
            Note: In order to separate from iou=0, function WILL return NaN if both 
            y_true and y_pred are 0. Need further treatment to remove nan in either 
            loss function or matrix.
        """
        super(IoU, self).__init__()
        assert mode in ['iou', 'dice']
        self.factor = {'iou': -1.0, 'dice': 0.0}[mode]
        self.eps = eps
        self.axis = axis
    
    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        sum_axis = list(range(1, self.axis)) + list(range(self.axis+1, y_pred.ndim))
        
        prod = (y_true * y_pred).sum(sum_axis)
        plus = (y_true + y_pred).sum(sum_axis)
        
        ## We keep nan for 0/0 in order to correctly apply weight
        iou = (2 + self.factor) * prod / (plus + self.factor * prod + self.eps)
        # print([y_true.shape, y_pred.shape, prod.shape, plus.shape])
        # print([prod, plus, iou])
        
        return iou


class SoftDiceLoss(IoU):
    def __init__(self, weight=None, ignore_index=[], reduction='mean',
                 mode='dice', axis=1, eps=0., use_positive=False):
        super(SoftDiceLoss, self).__init__(mode, axis, eps)
        self.ignore_index = ignore_index
        self.register_buffer('weight', weight)
        self.use_positive = use_positive
        self.reduction = {
            'none': lambda x: x,
            'mean': torch.mean,
            'sum': torch.sum,
        }[reduction]
    
    def _apply_weight(self, x):
        """ Apply class_weights to calculate loss, ignore nan. """        
        if self.weight is None:
            weight = torch.ones(x.shape[-1], device=x.device)
        else:
            weight = self.weight
        
        ## remove ignore_index
        idx = np.ones(x.shape[-1], dtype=bool)
        idx[self.ignore_index] = False
        x, weight = x[:,idx], weight[idx]
        
        ## apply weight
        weight = ~torch.isnan(x) * weight
        return x * weight / weight.sum(-1, keepdim=True)
    
    def forward(self, y_pred, y_true):
        ## y_pred is softmax cannot be 0, so no nan in res
        iou = super(SoftDiceLoss, self).forward(y_pred, y_true)
        # iou = torch.where(torch.isnan(iou), torch.zeros_like(iou), iou)
        iou = self._apply_weight(iou)
        # print(["apply_weights", res])
        res = -self.reduction(iou.sum(-1))
        
        return (res + 1) if self.use_positive else res
