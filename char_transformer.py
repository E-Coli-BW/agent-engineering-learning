"""
Mini Char-GPT: 字符级 Transformer 语言模型
==========================================
在 Mac 上可完整训练 + 推理 + 可视化的端到端示例

功能:
  1. 手写 Transformer Decoder (基于上一个文件的 MultiHeadAttention)
  2. 字符级语言建模 (Shakespeare / 自定义文本)
  3. 训练 Loss 曲线可视化
  4. Attention 热力图可视化
  5. 自回归文本生成
  6. 自动使用 MPS (Apple Silicon GPU) 加速
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import urllib.request
from dataclasses import dataclass

# ============================================================
# 0. 配置
# ============================================================
@dataclass
class GPTConfig:
    """模型和训练的超参数"""
    # --- 数据 ---
    block_size: int = 128       # 上下文窗口长度
    # --- 模型 ---
    vocab_size: int = 256       # 字符集大小 (ASCII)
    n_layer: int = 4            # Transformer 层数
    n_head: int = 4             # 注意力头数
    n_embd: int = 128           # embedding 维度
    dropout: float = 0.1
    # --- 训练 ---
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 3000
    eval_interval: int = 100    # 每隔多少步评估一次
    eval_iters: int = 20        # 评估时用多少 batch 估算 loss


# ============================================================
# 1. 设备选择 (自动检测 MPS / CUDA / CPU)
# ============================================================
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ============================================================
# 2. 数据准备
# ============================================================
class CharDataset:
    """字符级数据集：下载莎士比亚文本并编码"""

    DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    def __init__(self, block_size: int, data_dir: str = "data"):
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, "shakespeare.txt")

        # 下载数据
        if not os.path.exists(filepath):
            print(f"📥 下载莎士比亚文本...")
            urllib.request.urlretrieve(self.DATA_URL, filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # 构建字符级词表
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # char → int
        self.itos = {i: ch for i, ch in enumerate(chars)}  # int → char

        # 编码全部文本
        data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

        # 90% 训练, 10% 验证
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        self.block_size = block_size

        print(f"📊 数据统计:")
        print(f"   总字符数: {len(data):,}")
        print(f"   词表大小: {self.vocab_size}")
        print(f"   训练集: {len(self.train_data):,} 字符")
        print(f"   验证集: {len(self.val_data):,} 字符")

    def encode(self, s: str) -> list[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def get_batch(self, split: str, batch_size: int, device: torch.device):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)


# ============================================================
# 3. 模型组件
# ============================================================

class CausalSelfAttention(nn.Module):
    """因果自注意力 (手写版, 不用 nn.MultiheadAttention)"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.d_k = config.n_embd // config.n_head

        # QKV 合并成一个线性层 (效率更高, 面试加分项)
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 注册 causal mask 为 buffer (不是参数, 但会随模型保存)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

        # 用于可视化: 保存最近一次的 attention weights
        self.last_attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Step 1: 一次性计算 Q, K, V
        qkv = self.qkv_proj(x)  # (B, L, 3*D)
        Q, K, V = qkv.chunk(3, dim=-1)  # 各 (B, L, D)

        # Step 2: 拆多头
        Q = Q.view(B, L, self.n_head, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        K = K.view(B, L, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.n_head, self.d_k).transpose(1, 2)

        # Step 3: Scaled Dot-Product Attention
        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Step 4: Causal mask
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :L, :L] == 0, float("-inf"))

        # Step 5: Softmax + Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        self.last_attn_weights = attn_weights.detach()  # 保存用于可视化
        attn_weights = self.attn_dropout(attn_weights)

        # Step 6: 加权求和
        out = attn_weights @ V  # (B, H, L, d_k)

        # Step 7: 合并多头 + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.resid_dropout(self.out_proj(out))

        return out


class FeedForward(nn.Module):
    """Transformer 中的前馈网络 (两层 MLP + GELU)"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """一个 Transformer Decoder Block: LayerNorm → Attention → LayerNorm → FFN"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        # Pre-Norm 架构 (GPT-2 风格, 和原始 Transformer 的 Post-Norm 不同)
        x = x + self.attn(self.ln1(x))  # 残差连接
        x = x + self.ffn(self.ln2(x))   # 残差连接
        return x


class MiniGPT(nn.Module):
    """完整的 Mini GPT 模型"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享: embedding 和 output head 共享权重 (节省参数)
        self.token_emb.weight = self.lm_head.weight

        # 初始化
        self.apply(self._init_weights)
        print(f"🧠 模型参数量: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, L = idx.shape
        assert L <= self.config.block_size, f"序列长度 {L} 超过 block_size {self.config.block_size}"

        # Token Embedding + Position Embedding
        tok_emb = self.token_emb(idx)                              # (B, L, D)
        pos_emb = self.pos_emb(torch.arange(L, device=idx.device)) # (L, D)
        x = self.drop(tok_emb + pos_emb)

        # N 层 Transformer Block
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, L, vocab_size)

        # 计算 loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int = 40):
        """自回归生成"""
        self.eval()
        for _ in range(max_new_tokens):
            # 截断到 block_size
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-K 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        self.train()
        return idx


# ============================================================
# 4. 训练 + 可视化
# ============================================================
def train():
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端, 直接保存图片
    import matplotlib.pyplot as plt
    import numpy as np

    config = GPTConfig()
    device = get_device()
    print(f"\n🖥️  使用设备: {device}")

    # ---- 数据 ----
    dataset = CharDataset(block_size=config.block_size)
    config.vocab_size = dataset.vocab_size

    # ---- 模型 ----
    model = MiniGPT(config).to(device)

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # ---- 训练记录 ----
    train_losses = []
    val_losses = []
    steps = []

    @torch.no_grad()
    def estimate_loss():
        """用多个 batch 估算 loss (更稳定)"""
        model.eval()
        losses = {}
        for split in ["train", "val"]:
            batch_losses = []
            for _ in range(config.eval_iters):
                x, y = dataset.get_batch(split, config.batch_size, device)
                _, loss = model(x, y)
                batch_losses.append(loss.item())
            losses[split] = np.mean(batch_losses)
        model.train()
        return losses

    # ---- 训练循环 ----
    print(f"\n🚀 开始训练 ({config.max_iters} steps)...")
    print("-" * 60)

    model.train()
    for step in range(config.max_iters):
        # 定期评估
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss()
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            steps.append(step)
            print(f"  Step {step:5d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

        # 前向 + 反向
        x, y = dataset.get_batch("train", config.batch_size, device)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print("-" * 60)
    print("✅ 训练完成!")

    # ============================================================
    # 5. 可视化 1: Loss 曲线
    # ============================================================
    os.makedirs("outputs", exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(steps, train_losses, "b-o", label="Train Loss", markersize=3)
    ax.plot(steps, val_losses, "r-s", label="Val Loss", markersize=3)
    ax.set_xlabel("Training Steps", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title("Mini-GPT Training Curve (Character-Level Shakespeare)", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/training_loss.png", dpi=150)
    print(f"\n📈 Loss 曲线已保存: outputs/training_loss.png")

    # ============================================================
    # 6. 可视化 2: Attention 热力图
    # ============================================================
    model.eval()
    sample_text = "To be, or not to be"
    sample_ids = dataset.encode(sample_text)
    sample_tensor = torch.tensor([sample_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        _ = model(sample_tensor)

    # 取最后一层的 attention weights
    last_block = model.blocks[-1]
    attn_weights = last_block.attn.last_attn_weights  # (1, H, L, L)

    fig, axes = plt.subplots(1, config.n_head, figsize=(4 * config.n_head, 4))
    chars = list(sample_text)

    for h in range(config.n_head):
        ax = axes[h] if config.n_head > 1 else axes
        w = attn_weights[0, h].cpu().numpy()
        im = ax.imshow(w, cmap="viridis", aspect="auto")
        ax.set_title(f"Head {h}", fontsize=12)
        ax.set_xticks(range(len(chars)))
        ax.set_yticks(range(len(chars)))
        ax.set_xticklabels(chars, fontsize=7, rotation=90)
        ax.set_yticklabels(chars, fontsize=7)

    fig.suptitle(f'Attention Weights (Last Layer) — "{sample_text}"', fontsize=14)
    fig.tight_layout()
    fig.savefig("outputs/attention_heatmap.png", dpi=150)
    print(f"🔥 Attention 热力图已保存: outputs/attention_heatmap.png")

    # ============================================================
    # 7. 文本生成
    # ============================================================
    print("\n" + "=" * 60)
    print("📝 生成文本示例")
    print("=" * 60)

    prompts = [
        "ROMEO:",
        "To be, or not",
        "The king",
    ]

    for prompt in prompts:
        ids = dataset.encode(prompt)
        context = torch.tensor([ids], dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=200, temperature=0.8)
        text = dataset.decode(generated[0].tolist())
        print(f"\n--- Prompt: \"{prompt}\" ---")
        print(text[:300])
        print()

    # ============================================================
    # 8. 可视化 3: Embedding 空间 (PCA 降维)
    # ============================================================
    with torch.no_grad():
        embeddings = model.token_emb.weight.cpu().numpy()  # (vocab_size, n_embd)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    # 只标注可见字符
    fig, ax = plt.subplots(figsize=(12, 8))
    printable_indices = [i for i, ch in dataset.itos.items()
                         if ch.isprintable() and not ch.isspace()]

    ax.scatter(emb_2d[printable_indices, 0], emb_2d[printable_indices, 1],
               alpha=0.6, s=20, c="steelblue")

    # 标注部分有意义的字符
    highlight_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'")
    for i in printable_indices:
        ch = dataset.itos[i]
        if ch in highlight_chars:
            ax.annotate(ch, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=8, alpha=0.8)

    ax.set_title("Character Embedding Space (PCA)", fontsize=15)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/embedding_pca.png", dpi=150)
    print(f"🎨 Embedding PCA 图已保存: outputs/embedding_pca.png")

    print("\n" + "=" * 60)
    print("🎉 全部完成! 查看 outputs/ 文件夹获取可视化结果")
    print("=" * 60)


if __name__ == "__main__":
    train()
