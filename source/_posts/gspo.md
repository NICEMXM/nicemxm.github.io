---
title: GSPO：迈向持续拓展的语言模型强化学习
date: 2025-08-20
author: mxm
categories:
  - LLM
tags:
  - generate
---

## 引言
强化学习 （Reinforcement Learning，RL）已成为拓展语言模型、增强其深度推理与问题求解能力的关键技术范式。为了持续拓展 RL，首要前提是确保稳定、鲁棒的训练过程。然而，我们观察到现有的 RL 算法（如 GRPO）在长期训练中会暴露出严重的不稳定性问题并招致不可逆转的模型崩溃，阻碍了通过增加计算以获得进一步的性能提升。

为了能够持续拓展 RL，我们提出了 Group Sequence Policy Optimization (GSPO) 算法。不同于过去的 RL 算法，GSPO 定义了序列级别的重要性比率，并在序列层面执行裁剪、奖励和优化。相较于 GRPO，GSPO 在以下方面展现出突出优势：

- 强大高效：GSPO 具备显著更高的训练效率，并且能够通过增加计算获得持续的性能提升；
- 稳定性出色：GSPO 能够保持稳定的训练过程，并且根本地解决了混合专家（Mixture-of-Experts，MoE）模型的 RL 训练稳定性问题；
- 基础设施友好：由于在序列层面执行优化，GSPO 原则上对精度容忍度更高，具有简化 RL 基础设施的诱人前景。
以上优点促成了最新的 Qwen3 模型（Instruct、Coder、Thinking）的卓越性能。

## 序列级别的优化目标

设 x 为查询，πθold为用于采样回复的策略，{yi} i=1G为采样得到的回复组，^Ai为各个回复的组内相对优势，π θ为需优化的当前策略。GSPO 采用以下优化目标：

其中

这里的 si(θ) 即为 GSPO 基于序列似然定义的重要性比率，其中我们进行了长度归一化以降低方差并统一 si(θ) 的数值范围。

## 训练效率与性能
我们使用基于 Qwen3-30B-A3B-Base 微调得到的冷启动模型进行实验，并汇报其训练奖励曲线以及在 AIME'24、LiveCodeBench 和 CodeForces 等基准上的性能曲线。我们对比 GRPO 作为基线。注意 GRPO 必需采用 Routing Replay 训练策略才能正常收敛（我们将在后文讨论），而 GSPO 则无需该策略。

<img width="1802" height="1026" alt="image" src="https://github.com/user-attachments/assets/1245dd0a-6274-4a00-b9ff-fb89d023f146" />

从上图可见，GSPO 表现出比 GRPO 显著更高的训练效率，即在同等计算开销下能够取得更优的性能。特别地，我们观察到 GSPO 可以通过增加算力来获得持续的性能提升——这正是我们所期待的算法的可拓展性。最终，我们成功地将 GSPO 应用于最新的 Qwen3 模型的大规模 RL 训练，进一步释放了 RL scaling 的潜能！

一个有趣的观察是，GSPO 所裁剪的 token 比例比 GRPO 要高上两个数量级（如下图所示），但却具有更高的训练效率。这进一步表明 GRPO 采用的 token 级别的优化目标是有噪和低效的，而 GSPO 的序列级别的优化目标则提供了更可靠、有效的学习信号。

<img width="1580" height="410" alt="image" src="https://github.com/user-attachments/assets/7c46052a-ffe4-401c-977f-581e5f710fe5" />

## 对 MoE RL 和基础设施的收益
我们发现，当采用 GRPO 算法时，MoE 模型的专家激活波动性会使得 RL 训练无法正常收敛。为了解决这一挑战，我们过去采用了**路由回放（Routing Replay）**训练策略，即缓存 πθold中激活的专家，并在计算重要性比率时在 πθ中“回放”这些路由模式。从下图可见，Routing Replay 对于 GRPO 训练 MoE 模型的正常收敛至关重要。然而，Routing Replay 的做法会产生额外的内存和通信开销，并可能限制 MoE 模型的实际可用容量。

<img width="1782" height="572" alt="image" src="https://github.com/user-attachments/assets/bffd4ddc-4145-4c9e-bb96-67b459a9c2b6" />

GSPO 的一大突出优势在于彻底消除了对 Routing Replay 的依赖。其核心洞见在于：GSPO 仅关注序列级别的似然，而对个别 token 的似然不敏感。因此，其无需 Routing Replay 等对基础设施负担较大的手段，既简化和稳定了训练过程，又使得模型能够最大化地发挥容量与潜能。

此外，鉴于 GSPO 仅使用序列级别而非 token 级别的似然进行优化，直观上前者对精度差异的容忍度要高得多。因此，GSPO 使得直接使用推理引擎返回的似然进行优化成为可能，从而无需使用训练引擎进行重计算，这在 partial rollout、多轮 RL 以及训推分离框架等场景中特别有益。

## 结论
我们提出了 Group Sequence Policy Optimization (GSPO)，这是用于训练语言模型的全新 RL 算法。相较于 GRPO，GSPO 在训练稳定性、效率和性能方面展现出显著优势，并在 MoE 模型的大规模 RL 训练中表现出突出的功效。这些优点为最新 Qwen3 模型的卓越性能奠定了算法基础。以 GSPO 作为算法基石，我们将持续推动 RL scaling 的边界，并期待由此带来的智能进步。

