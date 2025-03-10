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

**估计KL散度**

KL散度是通过Schulman等人（2020）提出的近似器来估计的。近似器的定义如下：

$$D_{KL} [\pi_{\theta} \parallel \pi_{ref}] = \frac{\pi_{\theta}(o_{i,t} \mid q,o_{i,<t})}{\pi_{ref}(o_{i,t} \mid q,o_{i,<t})} - \log \left( \frac{\pi_{\theta}(o_{i,t} \mid q,o_{i,<t})}{\pi_{ref}(o_{i,t} \mid q,o_{i,<t})} \right) - 1,$$

**计算损失**

目标是最大化得分，同时确保模型保持接近参考策略。因此，损失定义如下：

$$L_{GRPO}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \left[ \frac{\pi_\theta(o_{i,t}|\mathbf{q}, o_{i,<t})}{[\pi_\theta(o_{i,t}|\mathbf{q}, o_{i,<t})]_{\text{no grad}}} \hat{A}_{i,t} - \beta D_{KL}[\pi_\theta \| \pi_{\text{ref}}] \right],$$

其中第一项代表缩放后的优势，第二项通过KL散度惩罚与参考策略的偏差。

在原始论文中，为了在每次生成后进行多次更新，利用了裁剪代理目标来推广这种表述方式：

$$L_{GRPO}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \left[\min\left(\frac{\pi_\theta(o_{i,t}|\mathbf{q}, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|\mathbf{q}, o_{i,<t})} \hat{A}_{i,t}, \text{clip}\left(\frac{\pi_\theta(o_{i,t}|\mathbf{q}, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|\mathbf{q}, o_{i,<t})}, 1-\epsilon, 1+\epsilon\right) \hat{A}_{i,t}\right) - \beta D_{KL}[\pi_\theta \| \pi_{\text{ref}}]\right],$$

这里的 $$\(\text{clip}(\cdot, 1-\epsilon, 1+\epsilon)\)$$ 确保更新不会过度偏离参考策略，通过将策略比率限制在$$\(1-\epsilon\)$$ 到 $$\(1+\epsilon\)$$ 之间。然而，在TRL中，如同原始论文一样，我们每次生成只做一次更新，因此可以简化损失至第一种形式。





