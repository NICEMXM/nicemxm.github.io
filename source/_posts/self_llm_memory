---
title: 大模型自身记忆模块进化-额外embedding层又是什么
date: 2026-10-16
author: mxm
categories:
  - LLM
tags:
  - generate
---

## transformer中的传统embedding层是什么？

transformer中的传统embedding层是通常称为Token Embedding

Token Embedding 在 Transformer 里的位置非常明确：**模型的最前端，第一个 Transformer Block 之前**。它不属于任何一个 Transformer Block 内部，而是整个模型的“入口层”。

### 📍 整体流程中的位置

以仅解码器（Decoder-only）的 GPT / Qwen 类模型为例：

```text
输入 token IDs
    ↓
Token Embedding 层        ← 就在这里
    ↓
得到 (batch, seq_len, hidden_size) 的向量
    ↓
加位置信息 / Dropout
    ↓
Transformer Block 1
    ↓
Transformer Block 2
    ↓
...
    ↓
Transformer Block N
    ↓
最终 LayerNorm
    ↓
LM Head（线性层）→ logits
```

所以，Token Embedding 是**所有注意力层和 FFN/MoE 层之前**的一步。

### 🧩 它具体做什么

Token Embedding 本质上是一个**查表操作**：

- 输入：离散的 token ID，形状 `(batch, seq_len)`
- 查表矩阵：形状 `(vocab_size, hidden_size)`
- 输出：每个 token 对应的稠密向量，形状 `(batch, seq_len, hidden_size)`

例如 Qwen3.5-35B-A3B：

- `vocab_size = 248,320`
- `hidden_size = 2048`
- Token Embedding 矩阵形状就是 `[248320, 2048]`

每个 token ID 会从这个矩阵中取出对应的一行，变成 2048 维向量，然后送入后面的 Transformer 层。

### 🔄 与位置编码的关系

在原始 Transformer 中：

```text
Input Embedding + Positional Encoding → Encoder/Decoder
```

位置编码是直接加到 Token Embedding 上的。

但在现代大模型（包括 Qwen 系列）中，常用 **RoPE（旋转位置编码）**。RoPE 不是在输入阶段加到 embedding 上，而是在**每一层自注意力内部**，对 Q 和 K 向量施加旋转。因此：

- Token Embedding 仍然在模型最前端；
- 位置信息注入的位置变了，但不影响 Token Embedding 的位置。

### 🧱 它和 Transformer Block 内部的关系

Transformer Block 通常包含：

1. 自注意力（Self-Attention）
2. FFN 或 MoE
3. 残差连接 + LayerNorm

Token Embedding 不在这些模块内部。它是第一层自注意力的输入。第一层自注意力会对这些 embedding 向量做线性变换，得到 Q、K、V，然后才开始注意力计算。

对于 MoE 模型，比如 Qwen3.5-35B-A3B：

- Token Embedding 是**稠密**的，所有 token 都会经过；
- MoE 的稀疏路由发生在 Transformer Block 内部的 FFN 部分；
- Token Embedding 本身不参与专家路由。

## 除了token embedding还有其它的么？

除了token embedding层，为了进一步增加模型记忆能力，还有如DeepSeek的**Engram模块**和Qwen3.8-Flash-Next的**N-gram Embedding**。

### 🧠 N-gram Embedding 

如果模型同时有 Token Embedding 和 N-gram Embedding，比如之前提到的 Qwen3.8-Flash-Next：

```text
token IDs
    ↓
Token Embedding  +  N-gram Embedding
    ↓
融合后的向量
    ↓
Transformer Block 1 ...
```

两者都在**输入阶段**，在第一个 Transformer Block 之前完成融合，然后一起送入主干网络。

### DeepSeek Engram Embedding 

DeepSeek Engram 是一个**条件记忆模块**，其核心思想是将大模型中“记忆”与“计算”这两种不同性质的负载解耦。它通过 O(1) 复杂度的哈希查找来直接检索静态知识，从而将宝贵的 GPU 计算资源解放出来，专注于更复杂的逻辑推理。

### 🧠 核心机制：如何实现“查表式”记忆？

Engram 的工作流程可以概括为“检索”与“融合”两个阶段，其设计非常精巧，确保检索到的记忆能被恰当地使用。

**第一阶段：确定性检索 (Retrieval)**
这个过程是将输入的局部上下文映射到静态的记忆表中，全程无需神经网络计算。

1.  **Tokenizer 压缩**：首先，模型会对输入的 token 进行归一化处理（如统一大小写、去除重音），将语义上等价的 token 映射到同一个 ID，从而压缩词表规模，提高记忆密度。
2.  **N-gram 切片与多头哈希**：对于序列中的每个位置，Engram 会提取其前面的 N-gram（例如 2-gram, 3-gram, 4-gram）。然后，使用**多个独立的哈希头**对每个 N-gram 进行哈希计算，每个哈希头负责索引记忆表中一个独立的“桶”区间，这大大降低了哈希冲突的概率。
3.  **O(1) 查表**：利用计算出的哈希值，直接从巨大的静态嵌入表中取出对应的向量行。这个过程是**确定性的**，时间复杂度为常数级 O(1)，与记忆表的总大小无关。

**第二阶段：上下文感知融合 (Fusion)**
检索到的记忆是静态的，可能带有噪声或与当前语境不符。因此，Engram 需要一个“门控”机制来决定是否采纳这些记忆。

- **门控机制 (Gating)**：模型会利用当前层的隐藏状态（已经过注意力机制，包含了全局上下文信息）生成一个**门控信号**。这个信号通过与检索到的记忆向量进行交互，计算出一个 0 到 1 之间的值。
- **残差融合**：如果检索到的记忆与当前上下文高度相关，门控值会趋近于 1，记忆被融入主干网络；反之，如果记忆与语境矛盾，门控值会趋近于 0，记忆被抑制，避免生硬注入。最终，门控后的记忆以**残差连接**的方式加到隐藏状态上。

### ⚙️ 在 DeepSeek-V4.1-Flash 中的具体实现

Engram 在 DeepSeek-V4.1-Flash 中得到了实际应用，其配置如下：

| 配置项 | 具体参数 |
| :--- | :--- |
| **Engram 层位置** | 模型的**第 1 层和第 14 层**（零索引） |
| **记忆表规模** | 每层约 **3.84 亿行**，条目为 256 字节的 FP8 格式，单层表权重约 **200 GB** |
| **总参数占比** | Engram 贡献了 **196B** 参数，与 552B 的主干参数共同构成模型 |
| **N-gram 阶数** | **{2, 3, 4}**，即同时使用 2-gram、3-gram 和 4-gram |
| **哈希头数量** | **8 个**哈希头，每个阶数的总嵌入维度为 2048 |
| **每头表大小** | 每个哈希头索引约 **1600 万**个条目 |
| **每 Token 访问量** | 每个 token 位置在每层请求 **24 行**，跨模型约 **12.4 KiB**，在 4 卡部署下每卡约 **3.1 KiB** |
| **推理内存占用** | 约 **189 GiB**，可通过 `--engram-config '{"cpu_offload":true}'` 卸载至主机内存 |

### 🔄 训练与推理的解耦设计

Engram 的设计哲学是“训练时联合，推理时解耦”。

- **训练阶段**：Engram 嵌入表与主干网络**联合端到端训练**。但由于其规模巨大（百亿级参数），单卡无法容纳，训练时采用 **All-to-All 通信**将嵌入表**分片（Shard）到多张 GPU** 上，让所有 GPU 协同更新。
- **推理阶段**：训练完成后，庞大的 Engram 表可以**完全卸载到主机内存（DRAM）甚至 SSD**。推理时，系统利用查找地址的**确定性**，通过后台 RDMA（远程直接内存访问）**异步预取**所需嵌入行，与 GPU 上的 Transformer 计算重叠进行，从而避免占用宝贵的 HBM 显存。

### ⚖️ 与 MoE 的互补关系

Engram 与 MoE（混合专家）是互补的稀疏化技术，共同构成了 DeepSeek 的“双重稀疏”架构。

| 对比维度 | **MoE（条件计算）** | **Engram（条件记忆）** |
| :--- | :--- | :--- |
| **稀疏化对象** | **计算路径**：每次只激活部分专家网络进行计算 | **存储路径**：每次只查找部分记忆表条目 |
| **扩展维度** | 通过增加专家数量扩展模型**容量与能力** | 通过增加记忆表规模扩展模型**知识储量** |
| **每 Token 成本** | 随激活专家数增加而增加 | **O(1)**，与记忆表总大小无关 |
| **角色定位** | 负责**动态推理**与**组合泛化** | 负责**静态知识**与**模式检索** |

DeepSeek 的论文指出，在固定的参数和算力预算下，将 **20% 至 25%** 的稀疏参数分配给 Engram 能获得最优的性能平衡，这被称为“**U 型扩展定律**”。

###  DeepSeek Engram模块和Qwen3.8-Flash-Next N-gram Embedding 的异同？

DeepSeek的**Engram模块**和Qwen3.8-Flash-Next的**N-gram Embedding**，**同属“条件记忆”这一大类的稀疏化技术，但并非完全相同**。Qwen的N-gram Embedding在设计上明确受到了DeepSeek Engram的启发，可视为其一种**更精简、面向工程落地的变体**。

它们的核心思想一致：都通过哈希查找N-gram来直接获取静态记忆，以O(1)的时间复杂度扩展模型的知识容量，并把记忆负载从主干的动态计算中剥离出来。

### 🧬 核心机制对比：同源异流

尽管核心思想一致，但两者在具体实现上体现了不同的设计哲学。

| 对比维度 | **DeepSeek Engram** | **Qwen3.8-Flash-Next N-gram Embedding** |
| :--- | :--- | :--- |
| **设计哲学** | **架构探索型**：追求更复杂、更精细的记忆机制，以充分释放“条件记忆”的潜力。 | **工程实用型**：追求极致的精简和效率，以最低的工程代价获得核心收益。 |
| **Token处理** | 压缩Tokenizer，合并语义相同的token，减少存储和哈希冲突。 | **直接哈希原始Token ID**，去掉压缩步骤，简化流程，避免信息损失。 |
| **注入位置** | **多个早期层**（如第2层和第15层），在不同深度注入记忆，实现更全面的增强。 | **单层**（第2层），在模型最早期注入，用最小的结构改动获取核心收益。 |
| **融合机制** | **上下文感知门控**：使用当前隐藏状态作为Query，动态决定是否采纳检索到的记忆，避免生硬注入。 | **门控融合**：同样采用门控机制，但整体结构更精简，目标是在最少参数下实现有效的条件记忆。 |
| **记忆表规模** | 在27B模型实验中，Engram记忆表参数约 **5.7B**，占总参数 **21.3%**。 | 在125B主模型外，额外配置 **51B** 的N-gram Embedding参数，占比更高。 |

### 💎 总结

DeepSeek Engram 像是一个**功能完备的“概念验证”架构**，它探索了条件记忆的多种可能性（如多层级注入、Token压缩、复杂的门控融合），为后续研究提供了宝贵的理论基础和设计范式。

Qwen的N-gram Embedding则是一个**高度工程化的“产品级”实现**。它直接借鉴了Engram的核心思想，但在工程上做了大量“减法”——去掉Token压缩、只在单层注入、简化融合机制——以最小的复杂度和计算开销，获得了条件记忆带来的主要收益（扩展容量、降低计算）。这种“精简”策略，使得51B的庞大嵌入表能够被有效地卸载到主机内存并进行预取，最终在125B MoE模型上实现了每token仅激活6B参数的高效率。

因此，它们是**同一技术路线上的不同实现**：Engram是更复杂的“探路者”，而Qwen的N-gram Embedding是更精简的“实践者”。
