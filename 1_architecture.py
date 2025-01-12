import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_norm = x / rms * self.weight
        return x_norm

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to q, k, v and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads, head_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class LLaMAModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_layers: int, num_heads: int, head_dim: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, head_dim)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, mask=None):
        x = self.token_emb(input_ids)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

# Example usage
if __name__ == "__main__":
    model = LLaMAModel(
        vocab_size=32000,
        dim=512,
        num_layers=6,
        num_heads=8,
        head_dim=64
    )
    
    # Test input
    x = torch.randint(0, 32000, (2, 128))  # (batch_size, seq_len)
    output = model(x)
    print(f"Output shape: {output.shape}")  # Expected: (2, 128, 32000)