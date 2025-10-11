import torch
from torch import nn
import numbers
from einops import rearrange, repeat
import torch.nn.functional as F


def pair(t):
    return t if isinstance(t, tuple) else (t, t)
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.Sigmoid(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Sigmoid(),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        # hidden_features = int(dim*ffn_expansion_factor)
        hidden_features = ffn_expansion_factor
        self.hid_fea = hidden_features
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        self.h, self.w = x.shape[2:]
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, endmember, fn):
        super(LayerNorm,self).__init__()
        self.norm = nn.LayerNorm(endmember)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

def minmax_normalization(tensor):
    min_val = tensor.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    max_val = tensor.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


class Spa(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, bias=False, LayerNorm_type='WithBias'):
        super(Spa, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Spa_Transformer(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_dim, bias)
        self.dim = dim
    def forward(self, x):
        self.h, self.w = x.shape[2:]
        x1 = x + self.attn(self.norm1(x))
        x2 = x1 + self.ffn(self.norm2(x1))
        return x2


class Spe(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, bias=False, LayerNorm_type='WithBias'):
        super(Spe, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Spe_Transformer(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_dim, bias)
        self.dim = dim

    def forward(self, x):
        self.h, self.w = x.shape[2:]
        x1 = x + self.attn(self.norm1(x))
        x2 = x1 + self.ffn(self.norm2(x1))
        return x2
