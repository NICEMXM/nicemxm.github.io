---
title: 深入了解 GRPO 方法
date: 2024-08-07
author: mxm
categories:
  - LLM
tags:
  - 强化学习
---

## 引言

GRPO 是一种在线学习算法，这意味着它通过使用训练模型本身在训练期间生成的数据进行迭代改进。GRPO 目标背后的直觉是最大限度地利用生成的完成，同时确保模型始终接近参考策略。要了解 GRPO 的工作原理，
可以分为四个主要步骤：采样生成、计算优势、估计 KL 散度和计算损失。
![image](https://github.com/user-attachments/assets/b446953d-c4b3-4ddb-852a-967605602873)

**采样生成**

在每个训练步骤中，我们都会对一批提示进行采样，并生成一组G每个提示的完成次数（表示为Oi）

**计算Reward**

对于每个G序列中，我们使用 reward 模型计算 Reward。为了与奖励模型的比较性质保持一致（通常在同一问题的输出比较数据集上进行训练），计算Reward以反映这些相对比较。它按如下方式规范化：

$$ \hat{A}_{i,t} = \frac{r_i - \text{mean}(r)}{\text{std}(r)} $$

这里的 $\text{clip}(\cdot, 1-\epsilon, 1+\epsilon)$ 确保更新不会过度偏离参考策略，通过将策略比率限制在$1-\epsilon$ 到 $1+\epsilon$ 之间。然而，在TRL中，如同原始论文一样，我们每次生成只做一次更新，因此可以简化损失至第一种形式。



