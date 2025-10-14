---
title: KV 缓存解释：优化 Transformer 推理效率
date: 2025-10-14
author: mxm
categories:
  - LLM
tags:
  - generate
---

## 介绍
当人工智能模型生成文本时，它们通常会重复许多相同的计算，这可能会减慢速度。KV缓存是一种通过记住前面步骤中的重要信息来帮助加快此过程的技术。该模型不是从头开始重新计算所有内容，而是重用它已经计算过的内容，从而使文本生成更快、更高效。

在这篇博文中，我们将以易于理解的方式分解 KV 缓存，解释它为什么有用，并展示它如何帮助 AI 模型更快地工作。

<img width="1401" height="739" alt="image" src="https://github.com/user-attachments/assets/74c829c7-f546-41bb-ba8a-714f159e14d6" />

##　先决条件
要充分掌握内容，读者应该熟悉：

* Transformer 架构：熟悉注意力机制等组件。
* 自回归建模：了解 GPT 等模型如何生成序列。
* 线性代数基础知识：矩阵乘法和转置等概念，对于理解注意力计算至关重要。

- 注意力权重的形状为[batch,h,Seq len ,Seq len ]]  
- 掩码多头注意力允许每个令牌由自身和所有先前的令牌表示。  
- 要生成一个新标记，模型需要查看所有以前的标记及其前面标记的表示

<img width="996" height="847" alt="image" src="https://github.com/user-attachments/assets/b97838a1-18e4-47bd-a354-18fa5f7a1182" />
<img width="1214" height="763" alt="image" src="https://github.com/user-attachments/assets/91650a3c-d7fe-4896-9412-08c8f811e764" />

https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/RsRm-SLIpIXdRwALshIB-.mp4

## 标准推理和 KV 缓存的兴起
当模型生成文本时，它会查看所有先前的标记以预测下一个标记。通常，它会对每个新代币重复相同的计算，这可能会减慢速度。

https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/PWI-EwqizVLInztmiI7Eo.mp4

- KV 缓存通过记住前面步骤中的这些计算来解决计算重叠问题，这可以通过在推理过程中存储注意力层的中间状态来实现。

https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/HnzDhoJdAbJhSassYjzEy.mp4

KV 缓存如何工作？
分步过程
* 第一步：当模型看到第一个输入时，它会计算并将其键和值存储在缓存中。
* Next Words：对于每个新单词，模型都会检索存储的键和值并添加新键和值，而不是重新开始。
* 高效的注意力计算：使用缓存的K和V与新的Q（query） 来计算输出。
* 更新输入：将新生成的令牌添加到输入中，并去返回自步2返回步骤 2直到我们完成生成。
<img width="3322" height="1320" alt="image" src="https://github.com/user-attachments/assets/c479814c-58ab-4ec4-b9f1-6bdd6228aee7" />

该过程如下所示：

```
Token 1: [K1, V1] ➔ Cache: [K1, V1]
Token 2: [K2, V2] ➔ Cache: [K1, K2], [V1, V2]
...
Token n: [Kn, Vn] ➔ Cache: [K1, K2, ..., Kn], [V1, V2, ..., Vn]

```

https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/DP2zDJTAU-yHrxVRh5GUt.mp4

https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/x0L80MqTJ4VPovbRY4yb2.mp4

在上表中，我们使用了dk=5为了获得更好的视觉效果，请注意，这个数字可能比我们展示的要大得多。

## 比较：KV 缓存与标准推理
以下是 KV 缓存与常规的比较：

|特征	|标准推理	|KV 缓存 |
|-----|---------|--------|
|每字计算	|该模型对每个单词重复相同的计算。	|该模型重用过去的计算以获得更快的结果。|
|内存使用情况	|每一步使用更少的内存，但内存会随着文本的延长而增加。	|使用额外的内存来存储过去的信息，但保持高效。|
|速度	|随着文本变长而变慢，因为它重复工作。	|避免重复工作，即使处理较长的文本也能保持快速。|
|效率	|计算成本高，响应时间慢。	|更快、更高效，因为模型会记住过去的工作。|
|处理长文本	|由于重复计算，难以处理长文本。	|非常适合长文本，因为它会记住过去的步骤。|

KV 缓存在速度和效率方面有很大差异，尤其是对于长文本。通过保存和重用过去的计算，它避免了每次都重新开始的需要，使其比生成文本的常规方式快得多。

## 实际实施
这是在 PyTorch 中实现 KV 缓存的简化示例：

```
# Pseudocode for KV Caching in PyTorch
class KVCache:
    def __init__(self):
        self.cache = {"key": None, "value": None}

    def update(self, key, value):
        if self.cache["key"] is None:
            self.cache["key"] = key
            self.cache["value"] = value
        else:
            self.cache["key"] = torch.cat([self.cache["key"], key], dim=1)
            self.cache["value"] = torch.cat([self.cache["value"], value], dim=1)

    def get_cache(self):
        return self.cache

```

使用 transformers 库时，默认情况下通过参数启用此行为，您还可以通过 cache_implementation 参数访问多种缓存方法，下面是一个极简的代码：use_cache

```
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B')
model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-1.7B').cuda()

tokens = tokenizer.encode("The red cat was", return_tensors="pt").cuda()
output = model.generate(
    tokens, max_new_tokens=300, use_cache = True # by default is set to True
)
output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

```

我们在 T4 GPU 上使用/不使用 kv 缓存对上面的代码进行了基准测试，我们得到了以下结果：

|使用 KV 缓存	|标准推理	|加速|
|-------------|---------|----|
|11.7 秒	|1 分钟 1 秒	|速度提高 ~5.21 倍|

## 结论

KV 缓存是一种简单但功能强大的技术，可帮助 AI 模型更快、更高效地生成文本。通过记住过去的计算而不是重复它们，它可以减少预测新单词所需的时间和精力。虽然它确实需要额外的内存，但此方法对于长时间对话特别有用，可确保快速高效的生成。
