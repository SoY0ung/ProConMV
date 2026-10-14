import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.cpp_extension import load

import math

class DropPath(nn.Module):
    """DropPath (Stochastic Depth)"""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x / keep_prob * binary_mask

# Load CUDA extension
T_MAX = 512 * 64
wkv_cuda = load(
    name="wkv",
    sources=["./models/cuda/wkv_op.cpp", "./models/cuda/wkv_cuda.cu"],
    verbose=True,
    extra_cuda_cflags=['-res-usage', '--maxrregcount 60', f'-DTmax={T_MAX}']
)


# CUDA-based WKV function
def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w, u, k, v)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B, ctx.T, ctx.C = B, T, C
        ctx.save_for_backward(w, u, k, v)

        y = torch.empty((B, T, C), device=k.device)
        wkv_cuda.forward(B, T, C,
                         w.float().contiguous(),
                         u.float().contiguous(),
                         k.float().contiguous(),
                         v.float().contiguous(),
                         y)
        return y.to(dtype=w.dtype)

    @staticmethod
    def backward(ctx, gy):
        B, T, C = ctx.B, ctx.T, ctx.C
        w, u, k, v = ctx.saved_tensors

        gw = torch.zeros((B, C), device=k.device)
        gu = torch.zeros((B, C), device=k.device)
        gk = torch.zeros((B, T, C), device=k.device)
        gv = torch.zeros((B, T, C), device=k.device)

        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)

        dtype = w.dtype
        return (None, None, None, gw.to(dtype).sum(dim=0), gu.to(dtype).sum(dim=0), gk.to(dtype), gv.to(dtype))


# Q-Shift spatial token shift
def q_shift(input, shift_pixel=1, gamma=1 / 4):
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    step = int(C * gamma)
    output[:, 0:step, :, shift_pixel:] = input[:, 0:step, :, :-shift_pixel]
    output[:, step:2 * step, :, :-shift_pixel] = input[:, step:2 * step, :, shift_pixel:]
    output[:, 2 * step:3 * step, shift_pixel:, :] = input[:, 2 * step:3 * step, :-shift_pixel, :]
    output[:, 3 * step:4 * step, :-shift_pixel, :] = input[:, 3 * step:4 * step, shift_pixel:, :]
    output[:, 4 * step:, :, :] = input[:, 4 * step:, :, :]
    return output


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, init_mode='fancy', key_norm=False):
        super().__init__()
        self.n_embd = n_embd
        self.recurrence = 1


        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(n_embd) if key_norm else None


        self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
        self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))

    def hilbert_curve(self, n, device=None, direction='horizontal'):
        """
    Generate Hilbert scan order recursively using matrix blocks.

    Returns:
        indices: torch.Tensor, linear coordinate of each visited point (row-major)
        order:   torch.Tensor, Hilbert scan visit order
    """
        def hilbert_matrix(k):
            if k == 1:
                return torch.tensor([[0, 3],
                                     [1, 2]], device=device)

            H = hilbert_matrix(k - 1)
            m = H.shape[0]
            N = m * m


            top_left  = H.T.clone()
            top_right = H + 3 * N
            bottom_left  = H + N
            bottom_right = torch.flipud(H) + 2 * N

            top = torch.hstack([top_left, top_right])
            bottom = torch.hstack([bottom_left, bottom_right])
            return torch.vstack([top, bottom])


        Hn = hilbert_matrix(n)
        N = Hn.shape[0]


        if direction == 'vertical':
            Hn = Hn.T
        elif direction != 'horizontal':
            raise ValueError("direction must be 'horizontal' or 'vertical'")



        coords = torch.argsort(Hn.reshape(-1))
        indices = coords.to(device)


        order = torch.argsort(indices)

        return indices, order


    def preprocess(self, x, resolution):
        """Shared preprocessing steps"""
        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = q_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

    def get_zigzag_indices(self, h, w, start='top-left', direction='horizontal', device=None):
        indices = []
        if start == 'top-left':
            row_start = 0
            col_start = 0
            row_step = 1
            col_step = 1 if direction == 'horizontal' else 1
        elif start == 'top-right':
            row_start = 0
            col_start = w - 1
            row_step = 1
            col_step = -1 if direction == 'horizontal' else -1
        elif start == 'bottom-left':
            row_start = h - 1
            col_start = 0
            row_step = -1
            col_step = 1 if direction == 'horizontal' else 1
        elif start == 'bottom-right':
            row_start = h - 1
            col_start = w - 1
            row_step = -1
            col_step = -1 if direction == 'horizontal' else -1

        for i in range(h):
            current_row = row_start + row_step * i
            if direction == 'horizontal':
                if current_row % 2 == 0:
                    cols = list(range(w))
                else:
                    cols = list(range(w - 1, -1, -1))
                for col in cols:
                    indices.append(current_row * w + col)
            elif direction == 'vertical':
                if (col_start + col_step * i) % 2 == 0:
                    rows = list(range(h))
                else:
                    rows = list(range(h - 1, -1, -1))
                for row in rows:
                    indices.append(row * w + (col_start + col_step * i))
        inv = indices.copy()
        for i, idx in enumerate(indices):
            inv[idx] = i
        return torch.tensor(indices, dtype=torch.long, device=device), torch.tensor(inv, dtype=torch.long, device=device)

    def forward(self, x, resolution):
        B, T, C = x.shape
        device = x.device
        h, w = resolution

        x_shifted = self.preprocess(x, resolution)


        r = torch.sigmoid(self.receptance(x_shifted))
        k_base = self.key(x_shifted)
        v_base = self.value(x_shifted)

        if self.spatial_decay.device != device:
            self.spatial_decay.data = self.spatial_decay.data.to(device)
            self.spatial_first.data = self.spatial_first.data.to(device)

        sv = torch.zeros_like(x_shifted,device=device)

        for direction in ['horizontal', 'vertical']:
            order, inverse = self.hilbert_curve(int(math.log2(h)), device, direction)
            k = k_base[:, order, :]
            v = v_base[:, order, :]

            for j in range(self.recurrence):
                decay = self.spatial_decay[j] / T
                first = self.spatial_first[j] / T
                v = RUN_CUDA(B, T, C, decay, first, k, v)

            if self.key_norm:
                v = self.key_norm(v)

            sv += v[:, inverse, :]

        return self.output(r * sv)


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, hidden_rate=4, init_mode='fancy', key_norm=False):
        super().__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, resolution):
        h, w = resolution
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = q_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        k = self.key(x)  # Kc
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)  # Vc
        x = torch.sigmoid(self.receptance(x)) * kv  # sRc [multiply] Vc

        return x


class VRWKV_Block(nn.Module):
    def __init__(self, n_embd, n_layer, hidden_rate=4, key_norm=False, drop_path=0.0):
        super().__init__()
        self.spatial_mix = VRWKV_SpatialMix(n_embd, n_layer, key_norm=key_norm)
        self.channel_mix = VRWKV_ChannelMix(n_embd, n_layer, hidden_rate=hidden_rate, key_norm=key_norm)

        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, resolution):
        # x: [B, C, H, W] → [B, H*W, C]
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Spatial + Channel mixing
        x = x + self.drop_path1(self.spatial_mix(self.norm1(x), resolution))
        x = x + self.drop_path2(self.channel_mix(self.norm2(x), resolution))

        # [B, H*W, C] → [B, C, H, W]
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class CrossVRWKV_Mix(nn.Module):
    def __init__(self, dim, recurrence=1, key_norm=False):
        super().__init__()
        self.dim = dim
        self.recurrence = recurrence

        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.key_norm = nn.LayerNorm(dim) if key_norm else None

        self.spatial_decay = nn.Parameter(torch.randn(recurrence, dim))
        self.spatial_first = nn.Parameter(torch.randn(recurrence, dim))

    def forward(self, img_feat, global_vec):
        """
        img_feat: [B, D]
        global_vec: [D]
        return: [B, D]
        """
        B, D = img_feat.shape
        device = img_feat.device

        # Expand global vector for batch
        global_vec = global_vec.unsqueeze(0).expand(B, -1)

        # Query-side modulation (receptance gate)
        r = torch.sigmoid(self.receptance(img_feat))  # [B, D]

        # Key / Value from global_vec
        k = self.key(global_vec)  # [B, D]
        v = self.value(global_vec)  # [B, D]

        # Recurrence (RWKV-style)
        out = v
        for j in range(self.recurrence):
            w = self.spatial_decay[j] / D
            u = self.spatial_first[j] / D
            # One-step WKV-like computation:
            # y = exp(w * 0) * v / exp(w * 0) = v (degenerates for single-step)
            # You may define a real recurrence across time or tokens in future
            out = RUN_CUDA(B, 1, D, w, u, k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)

        if self.key_norm:
            out = self.key_norm(out)

        return self.output(r * out)+ img_feat # gate fusion
