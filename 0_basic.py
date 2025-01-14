import torch
import torch.nn as nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

def rope_embedding(seq_len, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 2).float() / (dim // 2)))
    position_ids = torch.arange(seq_len).float().unsqueeze(1)
    freqs = position_ids @ inv_freq.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    return torch.cos(emb), torch.sin(emb)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    cos_part = cos[..., 0::2]
    sin_part = sin[..., 0::2]
    x_rotated_even = x_even * cos_part - x_odd * sin_part
    x_rotated_odd  = x_even * sin_part + x_odd * cos_part
    x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    x_out = x_out.flatten(-2)
    return x_out

seq_len = 10
dim = 8
batch_size = 2

cos, sin = rope_embedding(seq_len, dim)
print("cos shape:", cos.shape)
print("sin shape:", sin.shape)

x = torch.randn(batch_size, seq_len, dim)
print("x shape:", x.shape)

# 4. 对输入张量应用 RoPE
x_rotated = apply_rotary_pos_emb(x, cos, sin)
print("x_rotated shape:", x_rotated.shape)