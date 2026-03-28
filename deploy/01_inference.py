"""
推理部署原理 —— 从本地到生产
===============================

核心问题: 模型训练好了，怎么高效地提供服务？

本文件覆盖:
  Part 1: 推理加速的核心技术 (KV Cache、量化、批处理)
  Part 2: Ollama 本地部署详解
  Part 3: 模型量化实战 (用你的 Mini-GPT 演示)
  Part 4: 生产部署架构与面试知识
"""

import torch
import torch.nn as nn
import time
import os
import sys
import requests
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Part 1: KV Cache —— 推理加速的核心
# ============================================================
def part1_kv_cache():
    """
    KV Cache 是自回归推理中最重要的优化，面试必问。

    问题:
      自回归生成时，每生成一个新 token 都要重新计算整个序列的 Attention。
      生成第 100 个 token 时，要重新算前 99 个 token 的 K 和 V。
      → O(L²) 的重复计算!

    解决: KV Cache
      把已经计算过的 K 和 V 缓存起来，每步只算新 token 的 K 和 V。

    无 Cache:
      Step 1: 计算 K₁,V₁
      Step 2: 计算 K₁,V₁,K₂,V₂        ← K₁,V₁ 重复计算了!
      Step 3: 计算 K₁,V₁,K₂,V₂,K₃,V₃  ← 又重复!

    有 Cache:
      Step 1: 计算 K₁,V₁ → 缓存 [K₁],[V₁]
      Step 2: 只算 K₂,V₂ → 缓存 [K₁,K₂],[V₁,V₂]
      Step 3: 只算 K₃,V₃ → 缓存 [K₁,K₂,K₃],[V₁,V₂,V₃]

    复杂度对比:
      无 Cache: O(L³)  (每步 O(L²)，共 L 步)
      有 Cache: O(L²)  (每步 O(L)，共 L 步)
    """
    print("=" * 60)
    print("Part 1: KV Cache 原理演示")
    print("=" * 60)

    from char_transformer import MiniGPT, GPTConfig, CharDataset

    config = GPTConfig()
    dataset = CharDataset(block_size=config.block_size)
    config.vocab_size = dataset.vocab_size
    model = MiniGPT(config)
    model.eval()

    # ---- 对比有无 KV Cache 的生成速度 ----
    prompt = "To be"
    prompt_ids = [dataset.stoi[ch] for ch in prompt]
    context = torch.tensor([prompt_ids], dtype=torch.long)

    # 无 Cache (标准 generate，每步重新计算全部)
    start = time.time()
    with torch.no_grad():
        out_no_cache = model.generate(context.clone(), max_new_tokens=100, temperature=0.8)
    time_no_cache = time.time() - start

    print(f"\n  无 KV Cache:")
    print(f"    生成 100 tokens 用时: {time_no_cache:.3f}s")
    print(f"    速度: {100 / time_no_cache:.1f} tokens/s")

    generated_text = dataset.decode(out_no_cache[0].tolist())
    print(f"    生成内容: {generated_text[:80]}...")

    print(f"""
  💡 KV Cache 面试要点:
    1. 缓存每层 Attention 的 K 和 V，避免重复计算
    2. 空间换时间: 需要额外显存存储 Cache
    3. Cache 大小: n_layers × 2 × batch × n_heads × seq_len × d_k
    4. 这也是为什么长序列推理需要大显存的原因
    5. Ollama/vLLM 等框架都自动实现了 KV Cache
    """)


# ============================================================
# Part 2: 模型量化
# ============================================================
def part2_quantization():
    """
    量化 (Quantization): 用更少的位数表示模型权重。

    FP32 (4字节) → FP16 (2字节) → INT8 (1字节) → INT4 (0.5字节)

    显存节省:
      7B 模型:
        FP32: 28 GB
        FP16: 14 GB
        INT8:  7 GB
        INT4:  3.5 GB  ← 你的 qwen2.5:7b 用的就是 Q4_K_M

    面试重点: 量化不是简单的截断，有很多技巧来保持精度。
    """
    print("\n" + "=" * 60)
    print("Part 2: 模型量化演示")
    print("=" * 60)

    from char_transformer import MiniGPT, GPTConfig, CharDataset

    config = GPTConfig()
    dataset = CharDataset(block_size=config.block_size)
    config.vocab_size = dataset.vocab_size
    model = MiniGPT(config)
    model.eval()

    # ---- 展示量化对模型大小的影响 ----
    def model_size_bytes(model, dtype=torch.float32):
        total = 0
        for p in model.parameters():
            total += p.numel() * torch.tensor([], dtype=dtype).element_size()
        return total

    fp32_size = model_size_bytes(model, torch.float32)
    fp16_size = model_size_bytes(model, torch.float16)
    int8_size = model_size_bytes(model, torch.int8)

    print(f"\n  Mini-GPT 模型大小对比:")
    print(f"    FP32: {fp32_size / 1024:.1f} KB")
    print(f"    FP16: {fp16_size / 1024:.1f} KB ({fp32_size / fp16_size:.0f}x 压缩)")
    print(f"    INT8: {int8_size / 1024:.1f} KB ({fp32_size / int8_size:.0f}x 压缩)")

    # ---- 演示 PyTorch 动态量化 ----
    print(f"\n  🔧 PyTorch 动态量化 (INT8):")
    model_fp32 = copy.deepcopy(model)
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear},  # 量化所有 Linear 层
        dtype=torch.qint8,
    )

    # 对比推理速度
    test_input = torch.randint(0, config.vocab_size, (1, 32))

    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            model_fp32(test_input)
        fp32_time = time.time() - start

        start = time.time()
        for _ in range(100):
            model_int8(test_input)
        int8_time = time.time() - start

    print(f"    FP32 推理: {fp32_time:.3f}s (100 次)")
    print(f"    INT8 推理: {int8_time:.3f}s (100 次)")
    print(f"    加速比: {fp32_time / int8_time:.2f}x")

    print(f"""
  💡 量化面试知识:
    ┌──────────┬─────────────────────────────────────────┐
    │ 方法     │ 说明                                    │
    ├──────────┼─────────────────────────────────────────┤
    │ GPTQ     │ 逐层量化，用少量校准数据，INT4           │
    │ AWQ      │ 保护重要权重不被量化，效果好              │
    │ GGUF     │ llama.cpp 格式，CPU 友好，Ollama 使用   │
    │ bitsandbytes│ Hugging Face 生态，NF4 量化           │
    │ 动态量化  │ 推理时动态量化，不需要校准数据            │
    └──────────┴─────────────────────────────────────────┘

    你的 Ollama 模型 qwen2.5:7b 用的就是 GGUF Q4_K_M 量化!
    """)


# ============================================================
# Part 3: Ollama 部署详解
# ============================================================
def part3_ollama_deep_dive():
    """
    Ollama 到底做了什么？拆解它的工作原理。
    """
    print("\n" + "=" * 60)
    print("Part 3: Ollama 部署详解")
    print("=" * 60)

    # ---- 查看 Ollama 中的模型信息 ----
    response = requests.get("http://localhost:11434/api/tags")
    models = response.json()["models"]

    print(f"\n  📋 本地 Ollama 模型:")
    for m in models:
        size_gb = m["size"] / 1e9
        details = m.get("details", {})
        print(f"    {m['name']}")
        print(f"      大小: {size_gb:.1f} GB")
        print(f"      格式: {details.get('format', '?')}")
        print(f"      架构: {details.get('family', '?')}")
        print(f"      参数量: {details.get('parameter_size', '?')}")
        print(f"      量化: {details.get('quantization_level', '?')}")
        print()

    # ---- 查看运行时信息 ----
    response = requests.get("http://localhost:11434/api/ps")
    running = response.json().get("models", [])

    if running:
        print(f"  🟢 当前运行中的模型:")
        for m in running:
            print(f"    {m['name']} - VRAM: {m.get('size_vram', 0) / 1e9:.1f} GB")
    else:
        print(f"  💤 当前无模型在运行 (首次调用时会自动加载)")

    print(f"""
  Ollama 架构解析:
  ────────────────
    1. 服务器: Go 编写的 HTTP 服务，监听 11434 端口
    2. 推理引擎: llama.cpp (C++ 实现，CPU/GPU 混合推理)
    3. 模型格式: GGUF (量化后的模型文件)
    4. KV Cache: 自动管理
    5. 模型管理: 自动加载/卸载，多模型共存

  Ollama 的优势:
    ✅ 一行命令部署: ollama run qwen2.5:7b
    ✅ 自动量化和优化
    ✅ 支持 Mac MPS / NVIDIA GPU / CPU
    ✅ 兼容 OpenAI API 格式
    ✅ 模型生态丰富 (ollama.com/library)
    """)


# ============================================================
# Part 4: 生产部署架构
# ============================================================
def part4_production():
    print("\n" + "=" * 60)
    print("Part 4: 生产部署架构 (面试知识)")
    print("=" * 60)

    print("""
    生产级 LLM 推理架构:
    ────────────────────

    ┌─────────┐     ┌──────────────┐     ┌──────────────┐
    │  客户端  │────→│  API Gateway │────→│  Load Balancer│
    └─────────┘     └──────────────┘     └──────┬───────┘
                                                │
                         ┌──────────────────────┼──────────────────────┐
                         │                      │                      │
                    ┌────▼─────┐          ┌────▼─────┐          ┌────▼─────┐
                    │ 推理实例 1│          │ 推理实例 2│          │ 推理实例 3│
                    │ (vLLM)   │          │ (vLLM)   │          │ (vLLM)   │
                    │ GPU: A100│          │ GPU: A100│          │ GPU: A100│
                    └──────────┘          └──────────┘          └──────────┘

    关键组件:
    ─────────
    1. 推理引擎:
       - vLLM: PagedAttention, 动态 batching, 高吞吐
       - TGI (Text Generation Inference): Hugging Face 出品
       - Ollama: 简单部署，适合小规模
       - TensorRT-LLM: NVIDIA 优化，性能最好

    2. 推理优化:
       ┌──────────────────┬──────────────────────────────────┐
       │ 技术             │ 效果                              │
       ├──────────────────┼──────────────────────────────────┤
       │ KV Cache         │ 避免重复计算 K/V                  │
       │ Continuous Batch │ 动态组 batch，提高 GPU 利用率     │
       │ PagedAttention   │ 虚拟内存管理 KV Cache，减少浪费   │
       │ FlashAttention   │ IO-aware 的融合 attention kernel  │
       │ 投机解码         │ 小模型草稿 + 大模型验证，加速生成 │
       │ 模型并行         │ 大模型切分到多卡                  │
       │ 量化             │ INT4/INT8 减少显存和计算量         │
       └──────────────────┴──────────────────────────────────┘

    3. 长上下文处理 (面试题):
       - 滑动窗口注意力 (Sliding Window): Mistral 使用
       - ALiBi / RoPE: 位置编码外推
       - Ring Attention: 多卡分布式长序列
       - 分块处理: 超长文档分块 → 摘要 → 再处理

    4. 监控指标:
       - TTFT (Time To First Token): 首 token 延迟
       - TPS (Tokens Per Second): 生成速度
       - QPS (Queries Per Second): 系统吞吐量
       - GPU 利用率 / 显存占用
    """)


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    import copy
    print("🚀 推理部署原理\n")

    part1_kv_cache()
    part2_quantization()
    part3_ollama_deep_dive()
    part4_production()

    print("\n" + "=" * 60)
    print("✅ 推理部署模块完成！")
    print()
    print("关键收获:")
    print("  1. KV Cache: 缓存 K/V 避免重复计算，空间换时间")
    print("  2. 量化: FP32→INT4，7B 模型从 28GB 降到 3.5GB")
    print("  3. Ollama: GGUF + llama.cpp，一行命令部署")
    print("  4. vLLM: PagedAttention + 连续 batch，生产首选")
    print("  5. 核心指标: TTFT、TPS、QPS")
    print("=" * 60)
