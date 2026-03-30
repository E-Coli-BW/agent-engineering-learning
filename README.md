# 🧠 从零理解大模型：原理 → Agent → RAG

> **项目目标**：从最底层原理出发，逐步构建出完整的大模型应用技术栈。100% 本地运行，不依赖外部 API。

---

## 一、项目结构

```
agent-learning/
│
├── 📐 底层原理
│   ├── multi_head_attention.py   # 手写 QKV 注意力机制（面试核心）
│   └── char_transformer.py       # 完整 Mini-GPT 训练 + 可视化
│
├── 🤖 Agent 开发 (agent/)
│   ├── 01_chat_basics.py         # Level 1: LLM 调用的本质
│   ├── 02_tool_calling.py        # Level 2: 工具调用原理
│   ├── 03_react_agent.py         # Level 3: 手写 ReAct Agent
│   └── 04_langgraph_agent.py     # Level 4: LangGraph 生产级 Agent
│
├── 📚 RAG 系统 (rag/)
│   ├── 01_embedding_basics.py    # Level 1: Embedding 与向量检索本质
│   ├── 02_naive_rag.py           # Level 2: 手写最简 RAG
│   ├── 03_langchain_rag.py       # Level 3: LangChain 标准 RAG Pipeline
│   └── 04_advanced_rag.py        # Level 4: 高级 RAG (Multi-Query/CRAG/评估)
│
├── 🔧 LoRA 微调 (finetune/)
│   └── 01_lora.py                # 手写 LoRA + SFT + RLHF/DPO 原理
│
├── 🚀 推理部署 (deploy/)
│   └── 01_inference.py           # KV Cache + 量化 + Ollama + 生产架构
│
├── 🕸️ 知识图谱 (knowledge_graph/)
│   └── 01_graph_rag.py           # 手写知识图谱 + Graph RAG + 可视化
│
├── 🏗️ 工程化实战 (project/)
│   ├── config.py                 # 统一配置管理
│   ├── etl_pipeline.py           # ETL: 真实数据 Extract → Transform → Load
│   ├── api_server.py             # FastAPI RAG API (流式/健康检查/指标)
│   ├── mcp_server.py             # MCP Server (Copilot 工具集成)
│   ├── a2a_agent.py              # A2A Agent v1 (HTTP REST 问答)
│   ├── react_agent.py            # ReAct Agent v2 (LLM + 工具调用)
│   ├── wechat_bridge.py          # 微信 ↔ Agent 桥接器
│   └── app/                      # 生产版 package (模块拆分)
│
├── 🖥️ 前端 (frontend/)
│   ├── src/api.ts                # 统一 API 客户端
│   ├── src/pages/ChatPage.tsx    # 💬 Chat (RAG/A2A/ReAct 三模式切换)
│   ├── src/pages/DashboardPage.tsx # 📊 系统仪表盘
│   ├── src/pages/AgentsPage.tsx  # 🤖 Agent Card 管理
│   └── src/pages/ETLPage.tsx     # 🗄️ ETL 管道管理
│
├── 🚪 API Gateway (gateway/)
│   ├── pom.xml                   # Spring Cloud Gateway + Sentinel + Nacos
│   └── src/.../gateway/          # 路由/认证/限流/熔断/聚合健康检查
│
├── start.sh                      # 一键启动全栈 (5 个服务)
├── requirements.txt              # Python 依赖
└── README.md                     # 本文件
```

---

## 1.5 全栈架构

```
┌──────────────────────────────────────────────────────────────────┐
│  用户入口: Browser / 微信 / CLI / VS Code Copilot               │
└──────────────┬───────────────────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │  Java Gateway   │  :8080  Spring Cloud Gateway
       │  ┌─────────────┐│  • 统一路由: /api/rag → :8000
       │  │ Security    ││              /api/a2a → :5001
       │  │ Sentinel    ││              /api/react → :5002
       │  │ LoadBalancer││              /** → :3000 (前端)
       │  │ Logging     ││  • JWT 认证 (可选，默认关闭)
       │  └─────────────┘│  • Sentinel 限流/熔断
       └───┬──┬──┬──┬────┘  • 聚合健康检查 /health/all
           │  │  │  │
     ┌─────┘  │  │  └──────────────────────┐
     │        │  │                         │
┌────▼────┐ ┌─▼──▼──────┐ ┌──────────┐ ┌──▼─────────┐
│ Frontend│ │ RAG API    │ │ A2A Agent│ │ ReAct Agent│
│ :3000   │ │ :8000      │ │ :5001    │ │ :5002      │
│ React19 │ │ FastAPI    │ │ FastAPI  │ │ FastAPI    │
│ Vite    │ │ SSE stream │ │ A2A 协议  │ │ ReAct+Tools│
│ Tailwind│ │ /query     │ │ SSE 流式  │ │ 同步 only  │
└─────────┘ │ /health    │ │ /tasks/* │ │ /tasks/send│
            │ /metrics   │ └─────┬───┘ └──────┬─────┘
            │ /etl/run   │       │             │
            └──────┬─────┘       └──────┬──────┘
                   │                    │
             ┌─────▼─────┐      ┌──────▼──────┐
             │ ChromaDB   │      │   Ollama     │
             │ 向量库      │      │ :11434       │
             └───────────┘      │ qwen2.5:7b   │
                                │ mxbai-embed  │
                                └──────────────┘
```

### 服务清单

| 服务 | 端口 | 技术栈 | 职责 |
|------|------|--------|------|
| **API Gateway** | `:8080` | Java 21 + Spring Cloud Gateway + Sentinel | 统一入口、路由、认证、限流、熔断 |
| **RAG API** | `:8000` | Python + FastAPI + ChromaDB | 知识库问答 (向量检索 + LLM 生成) |
| **A2A Expert** | `:5001` | Python + FastAPI + Ollama | A2A 协议 Expert Agent (SSE 流式) |
| **ReAct Agent** | `:5002` | Python + FastAPI + Ollama | ReAct Agent + 工具调用 (同步) |
| **Frontend** | `:3000` | React 19 + Vite + Tailwind CSS | Chat & Manage UI |
| **Ollama** | `:11434` | Go (预装) | 本地 LLM 推理引擎 |

### 数据流示例

```
用户在前端输入: "LoRA 的原理是什么？"
  │
  ▼ Browser → http://localhost:8080 (Gateway)
  │           Sentinel 限流 → JWT 认证 → 请求日志 → 路由
  │
  ▼ Gateway 路由 /api/a2a/** → localhost:5001
  │
  ▼ A2A Agent: POST /tasks/sendSubscribe
  │            关键词匹配 → lora_finetuning skill
  │
  ▼ Ollama: /api/generate (stream=true)
  │         qwen2.5:7b 推理
  │
  ▼ SSE 流式返回 token → Gateway 透传 → 前端逐字显示
```

## 1.6 一键启动全栈

### 前置条件

| 依赖 | 版本 | 安装 |
|------|------|------|
| Python | 3.10+ | 运行后端服务 |
| Node.js | 18+ | 运行前端 |
| Java | 21+ | 运行 Gateway |
| Maven | 3.9+ | 编译 Gateway |
| Ollama | 0.1+ | 本地 LLM 推理 |

```bash
# 1. Ollama 运行 + 模型就绪
ollama serve &
ollama pull qwen2.5:7b && ollama pull mxbai-embed-large

# 2. Python 依赖
pip install langchain langchain-ollama langchain-community chromadb fastapi uvicorn

# 3. 前端依赖
cd frontend && npm install && cd ..

# 4. Gateway 编译 (首次启动会自动编译，也可手动)
cd gateway && mvn clean package -DskipTests && cd ..
```

### 启动 & 停止

```bash
./start.sh              # 一键启动全部 5 个服务
./start.sh status       # 查看所有服务状态
./start.sh stop         # 一键停止全部

# 单独启动某个服务:
./start.sh rag          # RAG API :8000
./start.sh a2a          # A2A Agent :5001
./start.sh react        # ReAct Agent :5002
./start.sh front        # 前端 :3000
./start.sh gateway      # Gateway :8080
./start.sh back         # 后端三件套 (RAG + A2A + ReAct)
```

### 启动顺序 (start.sh 内部)

```
1. RAG API Server    :8000  ← Python (需要 Ollama + ChromaDB)
2. A2A Expert Agent  :5001  ← Python (需要 Ollama)
3. ReAct Agent       :5002  ← Python (需要 Ollama)
4. Frontend Dev      :3000  ← Node.js (需要 npm install)
5. API Gateway       :8080  ← Java   (需要 mvn package)
```

### 访问方式

| 方式 | URL | 说明 |
|------|-----|------|
| **通过 Gateway** ⭐ | http://localhost:8080 | 统一入口，自动路由 |
| 直接访问前端 | http://localhost:3000 | Vite 开发服务器 |
| RAG API 文档 | http://localhost:8000/docs | FastAPI 自动文档 |
| 聚合健康检查 | http://localhost:8080/health/all | 一次看全部状态 |
| Gateway 路由表 | http://localhost:8080/actuator/gateway/routes | 当前路由 |

---

## 二、快速开始 (Setup Guide)

> 支持 **macOS** (Intel / Apple Silicon) 和 **Windows 10/11**，全程本地运行。

### 2.1 前置条件

| 依赖 | 最低版本 | 用途 |
|------|----------|------|
| Python | 3.10+ | 运行所有代码 |
| Git | 任意 | 克隆项目 |
| Ollama | 0.1.0+ | 本地 LLM 推理 |
| 磁盘空间 | ~10 GB | 模型文件 + 依赖 |
| 内存 (RAM) | ≥ 8 GB | 运行 7B 模型建议 16GB |

### 2.2 安装步骤

#### Step 1: 克隆项目

```bash
git clone <your-repo-url> agent-learning
cd agent-learning
```

#### Step 2: 创建 Python 虚拟环境

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### Step 3: 安装 Python 依赖

**方式 1: 一键安装 (推荐)**
```bash
pip install -r requirements.txt
```

**方式 2: 分步安装**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib scikit-learn
pip install langchain langchain-ollama langgraph langchain-community
pip install chromadb
```

> **Apple Silicon (M1/M2/M3/M4) 用户**：PyTorch 会自动支持 MPS 加速，无需额外配置。
>
> **有 NVIDIA GPU 的 Windows 用户**：将 `cpu` 换为 `cu121` 或 `cu124` 以启用 CUDA：
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

#### Step 4: 安装 Ollama + 下载模型

**macOS:**
```bash
# 方式 1: Homebrew
brew install ollama

# 方式 2: 官网下载
# 访问 https://ollama.com/download 下载 .dmg 安装
```

**Windows:**
```
# 访问 https://ollama.com/download 下载 Windows 安装包并安装
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

安装后，启动 Ollama 服务并下载模型：

```bash
# 启动服务 (macOS/Linux 后台运行，Windows 安装后自动启动)
ollama serve &

# 下载聊天模型 (~4.7 GB)
ollama pull qwen2.5:7b

# 下载 Embedding 模型 (~669 MB，RAG 模块需要)
ollama pull mxbai-embed-large
```

#### Step 5: 验证安装

```bash
# 验证 Python 环境
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import langchain; print(f'LangChain {langchain.__version__}')"

# 验证 Ollama
curl http://localhost:11434/api/tags
# 或
ollama list
```

### 2.3 运行顺序

按以下顺序逐步学习，每个模块都从底层原理到框架应用：

```bash
# ===== 📐 底层原理 =====
python multi_head_attention.py        # 1. 手写注意力机制
python char_transformer.py            # 2. Mini-GPT 训练 + 可视化 (⏱ ~5 min)

# ===== 🤖 Agent 开发 =====
python agent/01_chat_basics.py        # 3. LLM 调用本质
python agent/02_tool_calling.py       # 4. 工具调用原理
python agent/03_react_agent.py        # 5. 手写 ReAct Agent
python agent/04_langgraph_agent.py    # 6. LangGraph 生产级 Agent

# ===== 📚 RAG 系统 =====
python rag/01_embedding_basics.py     # 7. Embedding 与向量检索
python rag/02_naive_rag.py            # 8. 手写 RAG
python rag/03_langchain_rag.py        # 9. LangChain 标准 RAG
python rag/04_advanced_rag.py         # 10. 高级 RAG (Multi-Query/CRAG)

# ===== 🔧 LoRA 微调 =====
python finetune/01_lora.py            # 11. 手写 LoRA + SFT

# ===== 🚀 推理部署 =====
python deploy/01_inference.py         # 12. KV Cache + 量化 + 部署架构

# ===== 🕸️ 知识图谱 =====
python knowledge_graph/01_graph_rag.py  # 13. 知识图谱 + Graph RAG

# ===== 🏗️ 工程化实战 =====
python project/etl_pipeline.py          # 14. ETL Pipeline (真实数据处理)
python project/api_server.py            # 15. 启动 RAG API 服务
```

> **注意**：Agent / RAG / 知识图谱模块需要 Ollama 正在运行。底层原理和 LoRA 模块不需要 Ollama。

### 2.4 常见问题

<details>
<summary><b>Q: pip install 超时 / 下载慢？</b></summary>

使用国内镜像源：
```bash
pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple
```
或全局配置：
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
</details>

<details>
<summary><b>Q: Ollama 启动后 curl 连不上？</b></summary>

- macOS: 确保 Ollama 应用正在运行（任务栏有图标），或执行 `ollama serve`
- Windows: 确保 Ollama 在系统托盘运行
- 默认端口 `11434`，检查: `curl http://localhost:11434`
</details>

<details>
<summary><b>Q: Windows 上 PyTorch 安装失败？</b></summary>

确保 Python 版本为 3.10-3.12，并使用官方源：
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
</details>

<details>
<summary><b>Q: Mac 上 MPS 加速不生效？</b></summary>

需要 macOS 12.3+ 和 Apple Silicon (M1/M2/M3/M4)：
```python
import torch
print(torch.backends.mps.is_available())  # 应输出 True
```
Intel Mac 不支持 MPS，会自动回退到 CPU（不影响功能）。
</details>

<details>
<summary><b>Q: 7B 模型运行太慢 / 内存不够？</b></summary>

换用更小的模型：
```bash
ollama pull qwen2.5:3b    # 3B 版本，更快
ollama pull qwen2.5:1.5b  # 1.5B 版本，最轻量
```
然后修改代码中的 `CHAT_MODEL = "qwen2.5:3b"`。
</details>

<details>
<summary><b>Q: 可以用其他 LLM 吗？</b></summary>

可以。Ollama 支持众多模型，替换代码中的模型名即可：
```bash
ollama pull llama3.2:8b      # Meta LLaMA 3.2
ollama pull deepseek-r1:7b   # DeepSeek R1
ollama pull gemma2:9b        # Google Gemma 2
```
</details>

---

## 三、为什么要手写注意力机制？

在实际面试和研究中，调用 `nn.MultiheadAttention` 一行代码就能搞定注意力，但**你不理解它在做什么**。手写的价值在于：

| 维度 | 调库 | 手写 |
|------|------|------|
| 面试 | ❌ 无法回答追问 | ✅ 能说清每一步的 shape 变化 |
| 调试 | ❌ 黑盒，出错难定位 | ✅ 每一步可打断点检查 |
| 魔改 | ❌ 很难改内部逻辑 | ✅ 可以自由添加 sparse attention、Flash Attention 等 |
| 理解 | ❌ 知其然不知其所以然 | ✅ 真正理解为什么要除以 √d_k |

---

## 四、核心原理：注意力机制

### 4.1 直觉理解

注意力机制本质上是在回答一个问题：**序列中每个位置应该"关注"其他哪些位置？**

想象你在读一句话 `"The cat sat on the mat"`：
- 读到 `sat` 时，你的大脑会自动"注意"到 `cat`（谁在坐？）
- 读到 `mat` 时，你会"注意"到 `on`（在哪里？）

注意力机制用数学来模拟这个过程。

### 4.2 QKV 三元组

对于输入序列中的每个 token，我们通过三个不同的线性变换生成三个向量：

```
Q (Query)  = "我在找什么信息？"
K (Key)    = "我有什么信息可以被别人找到？"
V (Value)  = "如果你找到了我，我能提供什么内容？"
```

**类比图书馆**：
- 你带着一个**查询 (Q)**（"我想找关于猫的书"）
- 每本书有一个**索引标签 (K)**（"动物类"、"科技类"...）
- 每本书有实际**内容 (V)**
- 你用 Q 和所有 K 比较，找到最匹配的，然后取出对应的 V

### 4.3 Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

逐步拆解：

```
步骤 1: Q @ K^T
         计算每对 token 之间的"相似度分数"
         shape: (B, H, L, d_k) @ (B, H, d_k, L) → (B, H, L, L)
         结果是一个 L×L 的矩阵，[i,j] 表示 token_i 对 token_j 的关注程度

步骤 2: / √d_k
         缩放。为什么？
         如果 d_k 很大，点积的值会很大 → softmax 的输入值很大
         → softmax 输出接近 one-hot → 梯度消失
         除以 √d_k 让方差回到 1，保持梯度健康

步骤 3: mask (可选)
         对于自回归模型 (GPT)，token_i 不能看到 token_{i+1}, token_{i+2}...
         把上三角设为 -∞，softmax 后变成 0

步骤 4: softmax
         把分数归一化为概率分布
         每行加起来等于 1 → 权重分配

步骤 5: @ V
         用权重对 Value 做加权求和
         每个 token 获得一个"上下文感知"的新表示
```

### 4.4 Multi-Head Attention

**核心思想**：一个注意力头只能学到一种关注模式。多个头可以同时关注不同的东西。

```
Head 0: 可能学到"关注语法主语"
Head 1: 可能学到"关注前一个词"
Head 2: 可能学到"关注标点符号"
Head 3: 可能学到"关注语义相关词"
```

实现方式：
```
1. 把 d_model 维度均分成 H 份，每份 d_k = d_model / H
2. 每个头独立做 attention
3. 最后把所有头的输出拼接起来
4. 通过一个线性层 W_o 混合各头的信息
```

---

## 五、从注意力到 GPT：完整架构

### 5.1 Transformer Decoder Block

```
输入 x
  │
  ├──→ LayerNorm → Causal Self-Attention → + (残差连接)
  │                                        │
  ├────────────────────────────────────────←┘
  │
  ├──→ LayerNorm → Feed-Forward Network  → + (残差连接)
  │                                        │
  ├────────────────────────────────────────←┘
  │
输出
```

关键设计：
- **Pre-Norm**（GPT-2 风格）：先 LayerNorm 再做 attention，训练更稳定
- **残差连接**：让梯度可以"跳过"这一层直接回传，解决深层网络训练困难
- **FFN**：两层 MLP + GELU 激活，提供非线性变换能力（attention 本身是线性的！）

### 5.2 完整 GPT 模型

```
输入 token ids: [T, o, _, b, e]
        │
        ▼
Token Embedding: 每个字符 → 128 维向量
        +
Position Embedding: 每个位置 → 128 维向量
        │
        ▼
  ┌─────────────────┐
  │ Transformer ×4  │  ← 4 层叠加
  │ (Block 重复)     │
  └─────────────────┘
        │
        ▼
    LayerNorm
        │
        ▼
    Linear Head → 预测下一个字符的概率分布
```

### 5.3 训练目标

**Next Token Prediction（下一个 token 预测）**：

```
输入:  T  o  _  b  e
目标:  o  _  b  e  ,

对于每个位置，预测下一个字符是什么
损失函数: CrossEntropyLoss
```

这就是所有 GPT 模型的核心训练目标，从 GPT-1 到 GPT-4 都是如此，只是规模不同。

---

## 六、可视化解读

### 6.1 Loss 曲线 (`training_loss.png`)

```
Loss
 │\
 │ \___
 │      \___________  ← 收敛
 │                   
 └───────────────────→ Steps
```

- **Train Loss 持续下降** → 模型在学习
- **Val Loss 也下降** → 没有过拟合，泛化良好
- **Val Loss 开始上升** → 过拟合信号，需要 early stopping 或加 regularization

### 6.2 Attention 热力图 (`attention_heatmap.png`)

对 `"To be, or not to be"` 的注意力可视化：

- 颜色越亮 → 关注程度越高
- 对角线亮 → 关注自身（常见）
- 不同的 Head 展现不同模式 → 这就是 Multi-Head 的价值
- 可以观察到某些 Head 学会了"关注标点"或"关注重复词"

### 6.3 Embedding PCA (`embedding_pca.png`)

- 大小写字母倾向于聚在一起
- 数字可能形成独立的簇
- 标点符号分布在边缘
- 这说明模型学到了字符的语义相似性

---

## 七、面试常见追问

### Q: 为什么要除以 √d_k？
> 假设 Q 和 K 的每个元素是标准正态分布，它们点积的方差是 d_k。
> 除以 √d_k 使方差回归到 1，防止 softmax 进入饱和区导致梯度消失。

### Q: Multi-Head 比 Single-Head 好在哪？
> 单头只能学到一种注意力模式。多头可以在不同子空间学习不同的语义关系，
> 参数量相同（总维度不变），但表达能力更强。

### Q: Self-Attention 和 Cross-Attention 的区别？
> Self-Attention: Q=K=V=X（同一个序列内部互相关注）
> Cross-Attention: Q 来自 decoder，K/V 来自 encoder（跨序列关注）

### Q: Causal Mask 的作用？
> 在自回归生成时，token_i 不能看到 token_{i+1} 及之后的内容（未来信息泄露）。
> 通过下三角 mask 实现：上三角设为 -∞，softmax 后变为 0。

### Q: 为什么用 Pre-Norm 而不是 Post-Norm？
> Pre-Norm（GPT-2 风格）训练更稳定，不需要 warmup。
> Post-Norm（原始 Transformer）理论上收敛更好，但需要精心调参。

### Q: 参数量怎么算？
> Multi-Head Attention: 4 × (d_model² + d_model)（Q/K/V/O 四个线性层）
> FFN: 2 × (d_model × 4·d_model + 4·d_model)（两层 MLP）
> 一个 Block: 上述之和 + 2 个 LayerNorm

### Q: 时间/空间复杂度？
> Self-Attention: O(L² · d_model) 时间，O(L²) 空间
> 这是 Transformer 的核心瓶颈，也是 Flash Attention、线性注意力等工作的动机

---

## 八、在 Mac 上能做什么？

| 任务 | 可行性 | 建议 |
|------|--------|------|
| 字符级/词级语言模型 | ✅ 完全可以 | 本项目就是示例 |
| MNIST/CIFAR 图像分类 | ✅ 几分钟 | 经典入门任务 |
| 小型 Transformer (<10M 参数) | ✅ 用 MPS 加速 | 研究原理足够 |
| LoRA 微调小模型 | ✅ 适合 | GPT-2 Small 等 |
| 注意力变体研究 | ✅ 非常适合 | 快速验证想法 |
| 训练 LLaMA-7B+ | ❌ 显存不够 | 需要多卡 GPU |
| 大规模预训练 | ❌ 速度太慢 | 需要 A100 集群 |

**Apple Silicon (M1/M2/M3/M4) 通过 `torch.device("mps")` 可以获得 GPU 加速**，
对于研究原理和快速实验来说已经完全足够。

---

## 九、扩展方向

学完本项目后，可以继续探索：

1. **Grouped Query Attention (GQA)** — LLaMA 2/3 使用的注意力变体
2. **Flash Attention** — IO-aware 的高效注意力实现
3. **Rotary Position Embedding (RoPE)** — 替代绝对位置编码
4. **MultiAgent 多智能体协同** — 多 Agent 分工协作
5. **Neo4j 知识图谱** — 生产级图数据库
6. **vLLM / TGI** — 高吞吐推理引擎

---

## 十、项目完成度

| 模块 | 状态 | 核心文件 |
|------|------|----------|
| Transformer 手写 | ✅ 完成 | `multi_head_attention.py` |
| Mini-GPT 训练 + 可视化 | ✅ 完成 | `char_transformer.py` |
| Agent 开发 (4 级) | ✅ 完成 | `agent/01-04` |
| RAG 系统 (4 级) | ✅ 完成 | `rag/01-04` |
| LoRA 微调 + SFT/RLHF | ✅ 完成 | `finetune/01_lora.py` |
| 推理部署 | ✅ 完成 | `deploy/01_inference.py` |
| 知识图谱 + Graph RAG | ✅ 完成 | `knowledge_graph/01_graph_rag.py` |
| 工程化 RAG 系统 | ✅ 完成 | `project/etl_pipeline.py` + `project/api_server.py` |
| MCP Server | ✅ 完成 | `project/mcp_server.py` |
| A2A Agent 协作 | ✅ 完成 | `project/a2a_agent.py` |
| ReAct Agent + 工具 | ✅ 完成 | `project/react_agent.py` |
| 微信 Bot 桥接器 | ✅ 完成 | `project/wechat_bridge.py` |
| 前端 Chat & Manage | ✅ 完成 | `frontend/` (React + Vite + Tailwind) |
| Java API Gateway | ✅ 完成 | `gateway/` (Spring Cloud + Sentinel + Nacos) |
| 一键启动脚本 | ✅ 完成 | `start.sh` (5 服务生命周期管理) |

---

## 十一、微信 AI Bot (WeChat Bridge)

### 架构

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  微信用户 ←→ iLink Server ←→ WeChatBridge ←→ A2A Expert     │
│               (weixin.qq.com)   (桥接器)      (port 5001)    │
│                                    ↓                         │
│                              ExpertClient                    │
│                                    ↓                         │
│                         POST /tasks/send                     │
│                                    ↓                         │
│                           Ollama (qwen2.5:7b)                │
│                                                              │
│  核心组件:                                                    │
│  ├── wechat_bridge.py  — iLink 协议桥接，QR登录/收发消息      │
│  ├── a2a_agent.py      — Expert Agent (FastAPI + Ollama)     │
│  └── mcp_server.py     — MCP 工具层 (RAG/知识图谱/计算)      │
│                                                              │
│  特性:                                                       │
│  ├── ✅ QR 码扫码登录                                        │
│  ├── ✅ 私聊消息自动回复                                      │
│  ├── ✅ 多轮对话记忆 (每用户独立，最近5轮)                     │
│  ├── ✅ 技能路由 (Transformer/LoRA/RAG/Agent/推理/知识图谱)   │
│  ├── ✅ 长回答自动分段                                       │
│  ├── ✅ 打字指示器 ("对方正在输入...")                         │
│  ├── ✅ Session 持久化 (凭证+游标)                            │
│  └── ✅ 优雅退出 (Ctrl+C)                                    │
└──────────────────────────────────────────────────────────────┘
```

### 微信配置要求

你的微信需要能使用 **"爪机器人 (ClawBot)"** 的 iLink Bot API：
- 微信 → "发现" → "小程序" → 搜索 **"爪机器人"** 并授权开通
- 这是微信官方 iLink Bot API，非第三方逆向
- 仅支持**私聊**，不支持群聊
- 协议文档: https://www.wechatbot.dev/en/protocol

### 运行方式

```bash
# 终端 1: 启动 Expert Agent
python project/a2a_agent.py --serve

# 终端 2: 启动微信桥接器 (首次需扫码)
python project/wechat_bridge.py

# 流式模式
python project/wechat_bridge.py --stream

# 强制重新扫码
python project/wechat_bridge.py --login

# 连接远程 Expert
python project/wechat_bridge.py --expert http://192.168.1.100:5001
```

### 微信对话指令

| 指令 | 效果 |
|------|------|
| 任意问题 | AI 自动回答 |
| "清除历史" / "reset" | 清除多轮对话记忆 |

### 可选依赖

```bash
pip install qrcode   # 终端内显示 QR 码图案
```

---

## 十二、Agent 系统学习路径

本项目的 `project/` 目录包含了一个 Agent 系统从零到一的完整演进，
每一步都保留了代码，方便对比学习：

```
阶段 0 — "最简通信"
  agent/03_react_agent.py        教学版 ReAct Agent (理解原理)
  project/a2a_agent_v1_stdio.py  subprocess + stdin/stdout

阶段 1 — "标准化协议"
  project/a2a_agent.py           HTTP REST + SSE，对齐 A2A 规范
  project/wechat_bridge.py       微信 iLink 协议桥接器

阶段 2 — "LLM 自主决策"
  project/react_agent.py         ReAct Agent + 注册制工具调用
```

**阶段 0 → 1 学到的**: Agent 间通信从私有管道到标准 HTTP 协议的差距——
服务发现 (Agent Card)、能力声明 (Skills)、状态管理 (Task 生命周期)。

**阶段 1 → 2 学到的**: "纯问答"与"能执行动作的 Agent"的本质区别——
v1 靠 if/else 关键词路由，v2 让 LLM 自己决定调什么工具。
这就是从"对话机器人"到"AI Agent"的关键跨越。

**详细的踩坑笔记和设计决策**: 见 [`project/README.md`](project/README.md)

---

*Created with ❤️ for understanding Transformers from scratch.*
