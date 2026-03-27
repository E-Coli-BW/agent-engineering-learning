"""
手写 Multi-Head Self-Attention (面试版)
========================================
包含:
  1. Scaled Dot-Product Attention
  2. Multi-Head Attention
  3. 支持 causal mask (用于 decoder / GPT)
  4. 简单的测试用例
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. Scaled Dot-Product Attention (核心公式)
# ============================================================
def scaled_dot_product_attention(
    query: torch.Tensor,   # (B, H, L_q, d_k)
    key: torch.Tensor,     # (B, H, L_k, d_k)
    value: torch.Tensor,   # (B, H, L_k, d_v)
    mask: torch.Tensor = None,  # (B, 1, L_q, L_k) or (1, 1, L_q, L_k)
    dropout: nn.Dropout = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算公式: Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Returns:
        output: (B, H, L_q, d_v)  加权后的值
        attn_weights: (B, H, L_q, L_k)  注意力权重（可用于可视化）
    """
    d_k = query.size(-1)

    # Step 1: Q @ K^T → (B, H, L_q, L_k)
    attn_scores = torch.matmul(query, key.transpose(-2, -1))

    # Step 2: 缩放，防止点积过大
    attn_scores = attn_scores / math.sqrt(d_k)

    # Step 3: 应用 mask（把需要屏蔽的位置设成 -inf，softmax 后变成 0）
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

    # Step 4: Softmax 归一化 → 注意力权重
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Step 5: 可选 dropout（训练时随机丢弃一些 attention）
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # Step 6: 加权求和 → (B, H, L_q, d_v)
    output = torch.matmul(attn_weights, value)

    return output, attn_weights


# ============================================================
# 2. Multi-Head Attention
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 模块

    Args:
        d_model: 模型维度（输入/输出维度）
        num_heads: 注意力头数
        dropout: dropout 概率
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # ---- 核心参数：三个线性变换 + 输出线性变换 ----
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影
        self.W_k = nn.Linear(d_model, d_model)  # Key 投影
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        query: torch.Tensor,  # (B, L_q, d_model)
        key: torch.Tensor,    # (B, L_k, d_model)
        value: torch.Tensor,  # (B, L_k, d_model)
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Self-Attention: query = key = value = X
        Cross-Attention: query 来自 decoder, key/value 来自 encoder
        """
        batch_size = query.size(0)

        # ======== Step 1: 线性变换 ========
        Q = self.W_q(query)  # (B, L_q, d_model)
        K = self.W_k(key)    # (B, L_k, d_model)
        V = self.W_v(value)  # (B, L_k, d_model)

        # ======== Step 2: 拆成多头 ========
        # (B, L, d_model) → (B, L, H, d_k) → (B, H, L, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # ======== Step 3: 计算 Scaled Dot-Product Attention ========
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        # ======== Step 4: 合并多头 ========
        # (B, H, L_q, d_k) → (B, L_q, H, d_k) → (B, L_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # ======== Step 5: 输出线性变换 ========
        output = self.W_o(attn_output)  # (B, L_q, d_model)

        return output


# ============================================================
# 3. 生成 Causal Mask（用于 GPT 等自回归模型）
# ============================================================
def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """
    生成下三角 causal mask，防止看到未来的 token

    例如 seq_len=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Returns: (1, 1, L, L) 的 mask tensor
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)


# ============================================================
# 4. 测试
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # ---- 参数 ----
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8

    print("=" * 60)
    print("Multi-Head Self-Attention 手写实现测试")
    print("=" * 60)
    print(f"  Batch size:  {batch_size}")
    print(f"  Seq length:  {seq_len}")
    print(f"  d_model:     {d_model}")
    print(f"  Num heads:   {num_heads}")
    print(f"  d_k (每头):  {d_model // num_heads}")
    print()

    # ---- 构造输入 ----
    x = torch.randn(batch_size, seq_len, d_model)

    # ---- 不带 mask 的 Self-Attention ----
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.1)
    output = mha(query=x, key=x, value=x)
    print(f"[不带 mask] 输入 shape: {x.shape}")
    print(f"[不带 mask] 输出 shape: {output.shape}")
    print()

    # ---- 带 causal mask 的 Self-Attention (GPT 风格) ----
    causal_mask = generate_causal_mask(seq_len)
    output_masked = mha(query=x, key=x, value=x, mask=causal_mask)
    print(f"[Causal mask] 输出 shape: {output_masked.shape}")
    print(f"[Causal mask] mask shape: {causal_mask.shape}")
    print(f"[Causal mask] mask 示例 (4x4):")
    print(causal_mask[0, 0, :4, :4])
    print()

    # ---- 验证参数量 ----
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"总参数量: {total_params:,}")
    print(f"  = 4 × (d_model × d_model + d_model)")
    print(f"  = 4 × ({d_model} × {d_model} + {d_model})")
    print(f"  = 4 × {d_model * d_model + d_model}")
    print(f"  = {4 * (d_model * d_model + d_model):,} ✓")
    print()
    print("✅ 测试通过！")
