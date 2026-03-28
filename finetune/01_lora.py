"""
LoRA 微调原理 —— 从数学到实现
===============================

核心问题: 大模型动辄几十亿参数，全量微调需要巨大的显存和算力。
          有没有办法只调少量参数就能达到类似效果？

答案: LoRA (Low-Rank Adaptation) — 用低秩矩阵分解，
      只训练 0.1%~1% 的参数就能实现有效微调。

LoRA 原理 (面试必背!):
----------------------
假设原始权重矩阵 W ∈ R^(d×d)，LoRA 的核心假设是:

  微调时的权重变化 ΔW 是低秩的

即: ΔW = B × A  其中 B ∈ R^(d×r), A ∈ R^(r×d), r << d

  原始: y = Wx
  LoRA: y = Wx + BAx = (W + BA)x

参数量对比:
  全量微调: d × d (如 4096×4096 = 16M 参数)
  LoRA:     d × r + r × d = 2dr (如 r=8: 2×4096×8 = 65K 参数)
  压缩比:   16M / 65K ≈ 250x

为什么低秩假设合理？
  1. 预训练模型已经学到了很好的特征表示
  2. 微调只是小幅调整，变化量 ΔW 的有效维度很低
  3. 实验证明 r=8 或 r=16 就能达到接近全量微调的效果

本文件内容:
  Part 1: 手写 LoRA 层，理解数学原理
  Part 2: 把 LoRA 应用到我们的 Mini-GPT 上
  Part 3: 对比全量微调 vs LoRA 微调
  Part 4: 构造 SFT 数据集的方法论
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
import copy

# 让 import 找到项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Part 1: 手写 LoRA 层
# ============================================================
class LoRALinear(nn.Module):
    """
    手写 LoRA 层

    原始:  y = Wx + b
    LoRA:  y = Wx + b + (B @ A)x * (alpha / r)

    参数说明:
      - W: 原始权重 (冻结，不训练)
      - A: 降维矩阵 R^(r×d_in)，用高斯初始化
      - B: 升维矩阵 R^(d_out×r)，初始化为 0
      - alpha: 缩放系数 (控制 LoRA 的影响强度)
      - r: 秩 (rank)，越小参数越少

    为什么 B 初始化为 0？
      → 训练开始时 BA = 0，模型行为和原始一样
      → 保证微调的起点就是预训练模型
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # 缩放因子

        d_in = original_linear.in_features
        d_out = original_linear.out_features

        # ---- 冻结原始权重 ----
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # ---- LoRA 参数 (只有这两个矩阵被训练) ----
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.01)  # 高斯初始化
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))          # 零初始化!

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始路径 (冻结)
        original_output = self.original_linear(x)

        # LoRA 路径 (可训练)
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

        return original_output + lora_output

    def extra_repr(self):
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.2f}"


# ============================================================
# Part 2: 给模型注入 LoRA
# ============================================================
def inject_lora(model: nn.Module, target_modules: list[str], r: int = 8, alpha: float = 16.0) -> nn.Module:
    """
    遍历模型，把指定的 nn.Linear 层替换为 LoRALinear。

    target_modules: 要注入 LoRA 的模块名 (通常是 attention 的 Q/K/V/O)

    面试追问: 为什么通常只对 Attention 的线性层加 LoRA？
      → 因为 Attention 层占参数量大头
      → 实验证明对 Q 和 V 加 LoRA 效果最好
      → FFN 层的 LoRA 收益较小
    """
    lora_params = 0
    frozen_params = 0

    for name, module in model.named_modules():
        # 检查子模块中的 Linear 层
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear) and any(t in child_name for t in target_modules):
                lora_layer = LoRALinear(child_module, r=r, alpha=alpha)
                setattr(module, child_name, lora_layer)
                lora_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()

    # 统计参数
    for param in model.parameters():
        if param.requires_grad:
            pass  # 已统计
        else:
            frozen_params += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"  📊 LoRA 注入统计:")
    print(f"    总参数:     {total:,}")
    print(f"    可训练参数: {trainable:,}")
    print(f"    冻结参数:   {total - trainable:,}")
    print(f"    可训练比例: {trainable / total * 100:.2f}%")

    return model


# ============================================================
# Part 3: 在 Mini-GPT 上演示 LoRA 微调
# ============================================================
def demo_lora_on_minigpt():
    """
    把 LoRA 应用到我们在 char_transformer.py 中训练的 Mini-GPT 上。
    演示:
      1. 加载原始模型
      2. 注入 LoRA
      3. 对比参数量变化
      4. 用小数据集做 SFT 微调
    """
    print("=" * 60)
    print("Part 3: 在 Mini-GPT 上演示 LoRA")
    print("=" * 60)

    from char_transformer import MiniGPT, GPTConfig, CharDataset

    # ---- Step 1: 创建原始模型 ----
    config = GPTConfig()
    dataset = CharDataset(block_size=config.block_size)
    config.vocab_size = dataset.vocab_size
    model = MiniGPT(config)

    total_before = sum(p.numel() for p in model.parameters())
    print(f"\n  原始模型参数量: {total_before:,}")

    # ---- Step 2: 注入 LoRA ----
    print(f"\n  注入 LoRA (r=8, 目标: attention 的 QKV + 输出)...")
    model = inject_lora(
        model,
        target_modules=["qkv_proj", "out_proj"],  # 对 attention 层注入
        r=8,
        alpha=16.0,
    )

    # ---- Step 3: 准备 SFT 数据 ----
    # 模拟一个简单的指令微调场景:
    # 让模型学会在 "Q:" 后面生成 "A:" 格式的回答
    sft_data = """Q: What is attention?
A: Attention is a mechanism that allows the model to focus on relevant parts.

Q: What is a transformer?
A: A transformer is a neural network architecture based on self-attention.

Q: What is LoRA?
A: LoRA is a parameter-efficient fine-tuning method using low-rank matrices.

Q: What is GPT?
A: GPT is a generative pre-trained transformer for text generation.

Q: What is embedding?
A: Embedding maps discrete tokens into continuous vector representations.
"""

    # 编码 SFT 数据
    sft_encoded = torch.tensor(
        [dataset.stoi.get(ch, 0) for ch in sft_data],
        dtype=torch.long,
    )

    # ---- Step 4: LoRA 微调 ----
    device = torch.device("cpu")  # Mac 上用 CPU 就够了，数据量很小
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # 只优化 LoRA 参数!
        lr=1e-3,
    )

    print(f"\n  🚀 开始 LoRA 微调 (50 steps)...")
    model.train()
    losses = []
    block_size = config.block_size

    for step in range(50):
        # 随机采样一个训练片段
        if len(sft_encoded) > block_size + 1:
            idx = torch.randint(0, len(sft_encoded) - block_size - 1, (1,)).item()
        else:
            idx = 0
        x = sft_encoded[idx:idx + block_size].unsqueeze(0).to(device)
        y = sft_encoded[idx + 1:idx + block_size + 1].unsqueeze(0).to(device)

        if x.shape[1] < block_size or y.shape[1] < block_size:
            continue

        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 10 == 0:
            print(f"    Step {step:3d} | Loss: {loss.item():.4f}")

    print(f"  ✅ LoRA 微调完成！")
    print(f"    起始 Loss: {losses[0]:.4f}")
    print(f"    最终 Loss: {losses[-1]:.4f}")


# ============================================================
# Part 4: LoRA 变体与面试知识
# ============================================================
def lora_knowledge():
    print("\n" + "=" * 60)
    print("Part 4: LoRA 变体与面试知识")
    print("=" * 60)

    print("""
    ┌──────────────┬─────────────────────────────────────────────┐
    │ 方法         │ 说明                                        │
    ├──────────────┼─────────────────────────────────────────────┤
    │ LoRA         │ 低秩分解 ΔW = BA，冻结原始权重              │
    │ QLoRA        │ LoRA + 4-bit 量化原始权重 → 显存再降 4x     │
    │ AdaLoRA      │ 自适应分配不同层的 rank                      │
    │ LoRA+        │ 对 A 和 B 用不同学习率                       │
    │ DoRA         │ 分解权重为方向和幅度，分别用 LoRA 调         │
    │ 全量微调     │ 所有参数都训练 (baseline)                    │
    │ Prompt Tuning│ 只训练虚拟 prompt 的 embedding               │
    │ Adapter      │ 在 Transformer 层间插入小网络                │
    └──────────────┴─────────────────────────────────────────────┘

    面试必答:
    ─────────

    Q: LoRA 的原理是什么？
    A: 假设微调时权重变化 ΔW 是低秩的，将 ΔW 分解为 B×A 两个小矩阵。
       B 初始化为 0 保证起点等于预训练模型，A 用高斯初始化。
       推理时可以把 BA 合并回 W，不增加推理延迟。

    Q: r (rank) 怎么选？
    A: 通常 r=8 或 r=16。太小则表达能力不够，太大则失去参数高效的意义。
       复杂任务或领域差异大时可以适当增大。

    Q: LoRA 应用在哪些层？
    A: 通常应用在 Attention 的 Q、V 投影层效果最好 (原论文结论)。
       也可以加在 K、O 和 FFN 层，但边际收益递减。

    Q: QLoRA 和 LoRA 的区别？
    A: QLoRA = LoRA + NF4 量化 + 双重量化 + 分页优化器。
       将原始权重量化为 4-bit，再用 LoRA 在量化权重上做微调。
       显存需求从 LoRA 的 16-bit 降到 4-bit，几乎不损失效果。

    Q: LoRA 的推理性能影响？
    A: 训练时有额外的矩阵乘法 (BA)x。
       推理时可以把 BA 合并回原始权重 W' = W + BA，
       合并后和原始模型一样快，零额外开销。

    Q: SFT 数据怎么构造？
    A: 指令微调数据格式:
       {"instruction": "...", "input": "...", "output": "..."}
       质量 > 数量，千级别的高质量数据通常就够。
       数据来源: 人工标注、GPT-4 生成、已有任务数据转换。
    """)


# ============================================================
# Part 5: RLHF / DPO 概念速览
# ============================================================
def rlhf_dpo_overview():
    print("\n" + "=" * 60)
    print("Part 5: RLHF / DPO 概念速览 (面试必知)")
    print("=" * 60)

    print("""
    模型对齐三阶段 (OpenAI 方法论):
    ──────────────────────────────

    阶段 1: 预训练 (Pre-training)
      → 大量无标签数据上做 next token prediction
      → 获得强大的语言能力，但不一定"有用"或"安全"

    阶段 2: SFT (Supervised Fine-Tuning)
      → 用 (指令, 回答) 数据对做监督微调
      → 模型学会按指令格式回答问题
      → 但可能生成"正确格式但内容差"的回答

    阶段 3: RLHF (Reinforcement Learning from Human Feedback)
      → 步骤 1: 收集人类偏好 (回答 A 比 B 好)
      → 步骤 2: 训练 Reward Model (RM) 来模拟人类偏好
      → 步骤 3: 用 PPO 算法根据 RM 的奖励信号优化 LLM
      → 结果: 模型生成更符合人类期望的回答

    DPO (Direct Preference Optimization):
    ─────────────────────────────────────
      RLHF 的简化版本:
      → 跳过 Reward Model，直接用偏好数据优化 LLM
      → 将 RL 问题转化为分类问题
      → 训练更稳定、实现更简单
      → 效果接近 RLHF，已成为主流选择

    面试对比:
    ┌───────────┬─────────────────────────────┬──────────────────────────┐
    │           │ RLHF                        │ DPO                      │
    ├───────────┼─────────────────────────────┼──────────────────────────┤
    │ 复杂度    │ 高 (RM + PPO)               │ 低 (直接优化)            │
    │ 稳定性    │ PPO 训练不稳定               │ 训练更稳定               │
    │ 需要      │ Reward Model + 4 个模型      │ 只需 2 个模型            │
    │ 数据      │ 偏好数据 → RM → 在线采样     │ 只需要偏好数据           │
    │ 效果      │ 理论上限更高                  │ 接近 RLHF               │
    │ 代表      │ ChatGPT, GPT-4              │ LLaMA 2, Zephyr         │
    └───────────┴─────────────────────────────┴──────────────────────────┘
    """)


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    print("🚀 LoRA 微调: 从原理到实践\n")

    # Part 1 & 2 已在定义中展示

    # Part 3: 实际微调演示
    demo_lora_on_minigpt()

    # Part 4: 知识总结
    lora_knowledge()

    # Part 5: RLHF / DPO
    rlhf_dpo_overview()

    print("\n" + "=" * 60)
    print("✅ LoRA 微调模块完成！")
    print()
    print("关键收获:")
    print("  1. LoRA 核心: ΔW = BA，低秩分解，只训练 A 和 B")
    print("  2. B 初始化为 0 → 起点等于预训练模型")
    print("  3. 推理时可合并 W' = W + BA，零额外开销")
    print("  4. 通常对 Attention 的 Q/V 层注入效果最好")
    print("  5. QLoRA = LoRA + 4-bit 量化，进一步省显存")
    print("  6. RLHF → DPO 的演进: 简化对齐流程")
    print("=" * 60)
