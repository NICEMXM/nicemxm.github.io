---
title: Liger GRPO 遇见 TRL
date: 2025-06-16
author: mxm
categories:
  - TRL
tags:
  - generate
---

DR Liger 通过将内存使用量削减 40% 且模型质量零下降，增强了 TRL 的组相对策略优化 GRPO Trainer。我们还增加了对 FSDP 和 PEFT 的支持，使跨多个 GPU 扩展 GRPO 变得比以往任何时候都更容易。

### 动机
使用强化学习 （RL） 微调语言模型是模型训练生命周期中的关键步骤，用于引导模型朝着理想的行为发展，这些行为比通过典型的监督微调所能实现的要复杂。RL 传统上应用于使用近端策略优化 （PPO） 算法优化大型语言模型 （LLM）。这种方法通常与来自人类反馈的强化学习 （RLHF） 相关联，利用单独训练的奖励模型来指导主要模型的微调。

然而，使用 PPO 的 RLHF 是一种非常耗费资源的方法 - PPO 需要在内存中加载多个模型（策略、价值、奖励和参考模型），并且还需要对奖励和基础模型进行多次迭代才能达到预期的结果。RLHF 的成功还取决于奖励模型从我们的模型中有效区分期望行为和不期望行为的能力。

组相对策略优化 （GRPO） 最近与 DeepSeek 的 R1 模型一起受到了极大的欢迎。GRPO 避开了 RLHF 中使用的预训练奖励模型和价值模型，而是依赖于可验证的奖励函数，这些函数可以以封闭的方式检查模型输出的正确性，而无需外部奖励模型。当使用 GRPO 而不是 PPO 对易于验证的领域进行微调时，这带来了巨大的改进，例如教模型推理，并在数学和编码任务中表现良好。

下图显示了 GRPO 与 PPO 训练管道（参考：DeepSeekMath 的图 4：在开放语言模型中突破数学推理的极限）：

![image](https://github.com/user-attachments/assets/f72a0bf2-6774-4c07-8d57-273e7a2f7a5b)

也就是说，RL 训练仍然会消耗大量 GPU 内存，因此这里仍有很大的优化空间。在这篇博文中，我们讨论了我们最近添加到 TRL 的一项优化，该优化在 GRPO 训练期间将峰值内存使用量减少了 40%，我们还深入探讨了如何在不损失性能或正确性的情况下将 GRPO 扩展到多个 GPU 和节点。

### Liger Kernel 如何削减 GRPO 的内存

我们将 Liger 分块损失方法扩展到 GRPO 损失，这让我们不必为每个训练步骤将完整的 logit 存储在内存中。logits 的计算（涉及模型的输出头）是峰值内存使用的重要因素，尤其是在处理大词汇表、长序列长度或大批量时。我们通过将输入分块到整个批处理并一次运行一个块的正向传递来解决此问题。lm_head

但是，如果您只是以简单的方式实现它，您实际上将无法减少峰值内存，因为您仍然需要将所有 logits 保留在 GPU 内存中以进行向后传递。为了解决这个问题，我们在 forward pass 期间计算每个 loss chunk 的梯度（相对于 chunk 和 weight），然后在遍历每个 chunk 时累积它们。inputlm_head

以下是优化的可视化效果（参考：Byron Hsu）：

![image](https://github.com/user-attachments/assets/ba6beb57-a7f5-4e5e-b18d-bbe4588dae84)

与 TRL 的即插即用集成
我们最近在 PR #3184 中将 Liger GRPO 与 TRL 集成在一起，现在您只需在您的 中设置 即可使用 Liger GRPO 损失并享受内存节省！use_liger_lossTrueGRPOConfig

注意：这些功能尚未包含在最新的 TRL 版本中，因此您现在需要从源代码安装 TRL：

```
pip install "trl[liger] @ git+https://github.com/huggingface/trl.git"
```
然后你可以像这样使用它：

```
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset


train_dataset = load_dataset("trl-lib/tldr", split="train")
training_args = GRPOConfig(output_dir="Qwen3-0.6B-GRPO", use_liger_loss=True)

def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

### 基准

我们在有和没有 Liger GRPO Loss 的情况下进行了大量 GRPO 实验，以了解情况如何比较。对于策略模型，我们使用并尝试了不同的批处理大小。所有实验都使用其奖励函数在数据集上运行。Qwen3-0.6Bgsm8k

以下是 FP32 和 BF16 训练的峰值内存使用量与批处理大小的关系图。正如预期的那样，较大的批处理大小会更好地节省内存，因为我们沿批处理维度进行分块。因此，当批处理大小增加时，与常规（非 liger）版本相比，Liger 分块损失最终使用的内存要少得多，最多可减少 40%。

快速说明：目前，我们只支持 FP32，但我们正在努力为 TRL 中的 Liger GRPO 开源 BF16 支持。此处显示的 BF16 结果来自我们一直在测试的内部补丁。

![image](https://github.com/user-attachments/assets/69ce316d-bd09-4208-b56f-e4576063e8e9)

![image](https://github.com/user-attachments/assets/a3510b83-91b7-4310-b753-1c615bc20fdf)

我们还表明 Liger Loss 实际上是准确的。如图所示，训练步骤的奖励与您使用标准 TRL 实现看到的奖励几乎相同。


### 使用 FSDP 和 PEFT 进一步扩展

我们还分别在 PR #3260 和 PR #3355 中为 Liger GRPO Loss 添加了 FSDP 和 PEFT 支持，使用户能够轻松地在多个 GPU 或节点上扩展他们的实验。PEFT 技术（如 LoRA 和 QLoRA）通过仅在原始模型之上调整较小适配器权重的权重来减少可训练参数的数量，从而显著降低内存压力，因为整个模型的梯度、激活和优化器状态不需要保存在内存中。此外，在 GRPO 中使用 PEFT 可以避免在训练期间加载单独的参考模型，因为我们只需禁用 LoRA 适配器即可在训练期间获得原始的、未经修改的模型。

在这里，我们展示了一个使用 FSDP 和 PEFT 的多 GPU GRPO 训练图，其中我们比较了不同 Qwen3 模型大小中可能的最大训练批量大小，有和没有 Liger 损失。我们发现，使用 Liger，我们能够将批量大小提高大约 1.5 到 1.8 倍！

![image](https://github.com/user-attachments/assets/06942b52-2348-4597-95ed-0fa80b7e5d0f)

### 使用 vLLM 进一步扩展
为了在训练期间加速文本生成，Liger Loss 可以与 TRL 的集成 vLLM 服务器有效结合。这以最小的开销显著加快了数据的收集速度，并提供了无缝集成体验。

设置方法如下：

启动 vLLM 服务器：首先，启动 vLLM 服务器。此服务器将处理来自您的训练脚本的生成请求。打开终端并运行：

```
CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model "Qwen/Qwen3-0.6B"
```

注意：我们分配 CUDA_VISIBLE_DEVICES=1 以在特定 GPU（在本例中为 GPU 1）上运行 vLLM 服务器，让其他 GPU 免费进行训练。

配置并运行您的训练脚本：接下来，修改您的训练脚本以使用 vLLM 服务器。关键更改是 .use_vllm=TrueGRPOConfig

```
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset


def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

dataset = load_dataset("trl-lib/tldr", split="train[:1%]")
training_args = GRPOConfig(
    output_dir="Qwen3-0.6B-GRPO", 
    use_liger_loss=True, 
    use_vllm=True, # Enable vLLM integration
    logging_steps=10
)
trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B", # Ensure this matches the model served by vLLM
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

启动训练：最后，使用 （或者如果不使用 Accelerate 进行多 GPU/分布式训练） 运行训练脚本。如果您的 vLLM 服务器占用 GPU，请确保以不同的 GPU 为目标进行训练。accelerate launchpython

```
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py
```

通过执行这些步骤，您可以在 Liger Loss 的 GRPO 培训期间利用 vLLM 来加快生成周转时间。

### 结论
随着 Liger-GRPO 集成到 TRL 中，以及 FSDP 和 PEFT 支持，使用 GRPO 微调语言模型现在比以往任何时候都更加高效和可扩展。我们鼓励社区尝试这些新功能并分享他们的反馈，以帮助我们进一步改进 LLM 的 RL 训练。
