---
title: 直接偏好优化 （DPO）
date: 2025-02-11
author: mxm
categories:
  - LLM
tags:
  - RL
---

## 引言
直接偏好优化 （DPO） 提供了一种简化的方法，使语言模型与人类偏好保持一致。与需要单独的奖励模型和复杂的强化学习的传统 RLHF 方法不同，DPO 使用偏好数据直接优化模型。

## 了解 DPO
DPO 将偏好对齐重新构建为人类偏好数据的分类问题。传统的 RLHF 方法需要训练单独的奖励模型，并使用复杂的强化学习算法（如 PPO）来调整模型输出。**DPO 通过定义一个损失函数来简化此过程**，该损失函数根据首选输出和非首选输出直接优化模型的策略。

这种方法在实践中被证明非常有效，被用于训练 Llama 等模型。通过消除对单独奖励模型和强化学习阶段的需求，DPO 使偏好对齐更易于访问和稳定。

## DPO 的工作原理
DPO 过程需要监督微调 （SFT） 以使模型适应目标域。这通过在标准指令遵循数据集上进行训练，为偏好学习奠定了基础。该模型在保持其常规功能的同时学习基本任务完成情况。

接下来是偏好学习，其中模型在成对的输出上进行训练 - 一个首选和一个非首选。偏好对有助于模型了解哪些响应更符合人类价值观和期望。

DPO 的核心创新在于其直接优化方法。DPO 不是训练单独的奖励模型，而是使用二进制交叉熵损失来根据偏好数据直接更新模型权重。这种简化的流程使训练更加稳定和高效，同时获得与传统 RLHF 相当或更好的结果。

## DPO 数据集
DPO数据集通常通过将响应对标注为“首选”或“非首选”来创建。这可以手动完成，也可以使用自动过滤技术。以下是DPO单轮偏好数据集的示例结构：

| Prompt | Chosen | Rejected |
| --- | --- | --- |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |

“Prompt”列包含用于生成响应的提示。“Chosen”和“Rejected”列分别包含首选和非首选的响应。此结构有多种变体，例如包括系统提示列或包含参考材料的列。单轮对话的值可以表示为字符串，也可以表示为对话列表。

您可以在此处找到 Hugging Face 上的 DPO [数据集](https://huggingface.co/collections/argilla/preference-datasets-for-dpo-656f0ce6a00ad2dc33069478).


## 使用 TRL 实现
Transformers 强化学习 （TRL） 库使 DPO 的实施变得简单明了。和 类遵循相同样式的 API。 以下是设置 DPO 培训的基本示例：DPOConfigDPOTrainertransformers

```python
from trl import DPOConfig, DPOTrainer

# Define arguments
training_args = DPOConfig(
    ...
)

# Initialize trainer
trainer = DPOTrainer(
    model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    ...
)

# Train model
trainer.train()
```

## 最佳实践
数据质量对于成功实施 DPO 至关重要。偏好数据集应包括涵盖所需行为不同方面的不同示例。清晰的注释指南可确保首选和非首选响应的标记一致。您可以通过提高首选项数据集的质量来提高模型性能。例如，通过筛选较大的数据集以仅包含高质量的示例或与您的使用案例相关的示例。

在训练期间，仔细监控损失收敛并验证保留数据的性能。beta 参数可能需要调整，以平衡偏好学习与维护模型的常规功能。对不同的提示进行定期评估有助于确保模型在不过度拟合的情况下学习预期的偏好。

将模型的输出与参考模型进行比较，以验证首选项对齐的改进。对各种提示（包括边缘情况）进行测试有助于确保在不同场景中进行稳健的首选项学习。

后续步骤
⏩ 要获得 DPO 的实践经验，请尝试 DPO 教程。本实用指南将引导您使用自己的模型实现首选项对齐，从数据准备到训练和评估。

## DPO教程
整个DPO训练代码如下所示：
- 安装相关包
- 导入相关包
- 加载模型
- 设置DPO训练参数
- 开始训练
- 保存模型
- 推送模型到huggingface hub
  
```python
# 安装相关包
# Install the requirements in Google Colab
# !pip install transformers datasets trl huggingface_hub

# Authenticate to Hugging Face

from huggingface_hub import login

login()

# for convenience you can create an environment variable containing your hub token as HF_TOKEN

# 导入相关包
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

# 加载数据

# TODO: 🦁🐕 change the dataset to one of your choosing
dataset = load_dataset(path="trl-lib/ultrafeedback_binarized", split="train")

# TODO: 🦁 change the model to the path or repo id of the model you trained in [1_instruction_tuning](../../1_instruction_tuning/notebooks/sft_finetuning_example.ipynb)

# 选择并加载模型
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Model to fine-tune
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    torch_dtype=torch.float32,
).to(device)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Set our name for the finetune to be saved &/ uploaded to
finetune_name = "SmolLM2-FT-DPO"
finetune_tags = ["smol-course", "module_1"]

# DPO训练模型
# Training arguments
training_args = DPOConfig(
    # Training batch size per GPU
    per_device_train_batch_size=4,
    # Number of updates steps to accumulate before performing a backward/update pass
    # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
    gradient_accumulation_steps=4,
    # Saves memory by not storing activations during forward pass
    # Instead recomputes them during backward pass
    gradient_checkpointing=True,
    # Base learning rate for training
    learning_rate=5e-5,
    # Learning rate schedule - 'cosine' gradually decreases LR following cosine curve
    lr_scheduler_type="cosine",
    # Total number of training steps
    max_steps=200,
    # Disables model checkpointing during training
    save_strategy="no",
    # How often to log training metrics
    logging_steps=1,
    # Directory to save model outputs
    output_dir="smol_dpo_output",
    # Number of steps for learning rate warmup
    warmup_steps=100,
    # Use bfloat16 precision for faster training
    bf16=True,
    # Disable wandb/tensorboard logging
    report_to="none",
    # Keep all columns in dataset even if not used
    remove_unused_columns=False,
    # Enable MPS (Metal Performance Shaders) for Mac devices
    use_mps_device=device == "mps",
    # Model ID for HuggingFace Hub uploads
    hub_model_id=finetune_name,
    # DPO-specific temperature parameter that controls the strength of the preference model
    # Lower values (like 0.1) make the model more conservative in following preferences
    beta=0.1,
    # Maximum length of the input prompt in tokens
    max_prompt_length=1024,
    # Maximum combined length of prompt + response in tokens
    max_length=1536,
)

trainer = DPOTrainer(
    # The model to be trained
    model=model,
    # Training configuration from above
    args=training_args,
    # Dataset containing preferred/rejected response pairs
    train_dataset=dataset,
    # Tokenizer for processing inputs
    processing_class=tokenizer,
    # DPO-specific temperature parameter that controls the strength of the preference model
    # Lower values (like 0.1) make the model more conservative in following preferences
    # beta=0.1,
    # Maximum length of the input prompt in tokens
    # max_prompt_length=1024,
    # Maximum combined length of prompt + response in tokens
    # max_length=1536,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(f"./{finetune_name}")

# Save to the huggingface hub if login (HF_TOKEN is set)
if os.getenv("HF_TOKEN"):
    trainer.push_to_hub(tags=finetune_tags)
```
