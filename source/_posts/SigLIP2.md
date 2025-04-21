---
title: SigLIP2：更好的多语言视觉语言编码器
date: 2024-08-07
author: mxm
categories:
  - LLM
tags:
  - generate
---

Google 发布了一个更好的全新多语言视觉语言编码器系列 SigLIP 2。SigLIP 2 模型在所有模型尺度上的核心功能都优于旧的 SigLIP 模型，包括零样本分类、图像文本检索和为视觉语言模型 （VLM） 提取视觉表示时的性能。

![image](https://github.com/user-attachments/assets/502b3b41-016e-422a-9ebc-b852a5686cc3)

视觉编码器很简单 - 它们获取图像，将其编码为表示，该表示用于下游任务，如分类、对象检测、图像分割和更多视觉任务。研究人员一直在追求密集、局部感知和语义丰富的视觉表示。

CLIP 和 ALIGN 是通过联合训练将图像编码器和文本编码器对齐在一起的第一个示例。这种方法开辟了训练视觉模型的新方法。SigLIP 更进一步，用 sigmoid 损失取代了 CLIP 的对比损失，以获得更好的编码器。

凭借更智能的训练目标，我们不断构建更结构化、更精细、更强大的视觉编码器。SigLIP 2 就是这样，在 SigLIP 的基础上应用了一系列非常有趣和智能的训练目标，以提供更好、更强的视觉语言编码器。

我们将在这篇博文中尝试一些新的东西。我们不会说明什么是新的以及在哪里可以找到它，而是一起进行一些练习。我们从 SigLIP 开始，然后集思广益，提出一系列问题和答案（新标题），以逐渐涵盖 SigLIP 2 中的所有更新。听上去很好？

## 问题 1：我们可以使用什么（低成本）辅助训练目标来学习更好的视觉表征（在位置感知和局部性方面）？

添加一个解码器（就是这么简单）

让我们在系统中加入一个解码器。现在我们有一个图像编码器、一个文本编码器和一个文本解码器。文本解码器将有三个目标：

- 预测整体图像标题
- 根据描述特定图像区域的字幕预测边界框坐标
- 根据边界框坐标预测区域特定的字幕

解码器为视觉编码器提供了额外的信号，使其具备位置感知能力。这标志着 SigLIP 2 训练方案的首次改进。

## 问题 2：如何改进图像表征的细粒度局部语义？

**使用全局-局部损失和掩码预测进行自蒸馏**

为了改进图像表征中的细粒度局部语义，我们引入了两个关键训练目标：**全局-局部损失（Global-Local Loss）** 和 **掩码预测损失（Masked Prediction Loss）**。借鉴自监督学习的相关研究，我们采用**自蒸馏（Self-Distillation）**。我们可以使用同一个模型作为教师和学生，每次迭代时，教师模型的参数将是学生模型参数的移动平均值。

- **全局-局部损失（Global-Local Loss）**：学生网络仅接收训练图像的局部视图，并训练其表征，使其匹配教师网络从完整图像得出的表征。
- **掩码预测损失（Masked Prediction Loss）**：学生网络中 50% 的嵌入图像块被掩码标记覆盖，要求学生模型在这些掩码位置匹配教师模型的特征。

这些训练目标教会视觉编码器空间感知能力，并提高其局部语义表达能力。研究人员在**80% 的训练完成后**才引入这些损失，而在早期阶段主要采用 sigmoid 损失和解码器损失。这么做的原因是为了**节省计算资源**（额外的损失计算成本较高），同时**避免对编码器产生负面影响**。

## **问题 3：如何使模型适应不同的分辨率？**

### **适应不同的分辨率**
众所周知，图像模型对分辨率和长宽比的变化非常敏感。在这里，我们可以利用两种不同的方法来使这些模型适应不同的分辨率和补丁大小。

1. **固定分辨率变体（Fixed resolution variant）**：
   - 在 95% 训练完成后，我们可以调整**位置嵌入（positional embeddings）**和**补丁嵌入（patch embeddings）**的大小，然后继续训练，以适应所需（可能更大）的分辨率。

2. **动态分辨率变体（Dynamic resolution variant）**：
   - 受到 **FlexiViT**（可以使用不同序列长度的输入）和 **NaViT**（遵循原生长宽比）方法的启发，我们可以创建 **NaFlex** 变体。这种方法非常有趣，因为它可以在单一模型中适应 **OCR（较小的长宽比变形）** 和 **文档理解（适当的分辨率）**。
   - 具有 **"-naflex"** 后缀的模型是动态分辨率变体。相比之下，固定分辨率模型可以直接用于现有的类别，而动态分辨率变体则需要专门调用。
     
这标志着 SigLIP 到 SigLIP 2 发展的结束。接下来的部分，我们将探索 **SigLIP 2** 的应用场景。

**使用 Transformers 进行推理**

在这些模型上运行推理非常简单。你可以复制粘贴下面的代码，在 **免费版 Colab** 笔记本上运行推理 

要在 **SigLIP 2** 上运行推理，请从以下来源安装，或使用这个稳定分支：
```
pip install git+https://github.com/huggingface/transformers@v4.49.0-SigLIP-2
```

**零样本分类（Zero-shot Classification）**

在这里，我们使用**便捷的 API**来展示 **SigLIP 2** 的零样本分类能力。

```
from transformers import pipeline

ckpt = "google/siglip2-so400m-patch14-384"
pipe = pipeline(model=ckpt, task="zero-shot-image-classification")

inputs = {
    "images": [
        "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg", # bear
        "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000776.jpg", # teddy bear
    ],
    "texts": [
        "bear looking into the camera",
        "bear looking away from the camera",
        "a bunch of teddy bears",
        "two teddy bears",
        "three teddy bears"
    ],
}

outputs = pipe(inputs["images"], candidate_labels=inputs["texts"])

```

**让我们来可视化输出结果。**
![image](https://github.com/user-attachments/assets/6aababd9-8328-41e9-9c90-0b2513b43623)

**用于下游任务的图像编码**

你还可以使用以下方法对图像进行编码：

```
import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

ckpt = "google/siglip2-so400m-patch14-384"
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)

image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
inputs = processor(images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    image_embeddings = model.get_image_features(**inputs)    

print(image_embeddings.shape) # torch.Size([1, 1152])

```

**比较 SigLIP 1 与 SigLIP 2**

查看所有发布的 **SigLIP 2** 模型表，我们可以发现 **SigLIP** 相较于之前版本发生了两个显著变化：

1. **SigLIP 2** 引入了适用于 **动态分辨率** 的新变体（**naflex**）。
2. **SigLIP 2** 增加了 **1B 系列（giant）**。

**SigLIP 2** 的评估表明确展示了其相较 **SigLIP** 的卓越性能。

![image](https://github.com/user-attachments/assets/044d5ac4-6470-4ea3-bb84-5d9ba220916d)

**用于 VLM 的编码器**

与文本信息对齐的视觉编码器在 **视觉语言模型（VLMs）** 的开发中变得越来越重要。构建 **VLM** 的常见方法是将**预训练的视觉编码器**与**预训练的大型语言模型（LLM）**结合，并利用**多模态数据**在各种视觉-语言任务上共同训练。

一个值得关注的 **VLM** 例子是 **PaliGemma**，它充分利用了 **SigLIP** 视觉编码器系列。你可以在 **PaliGemma** 博客文章中深入了解其能力。在此基础上，**新推出的 PaliGemma 2** 更进一步，**将 SigLIP 与先进的 Gemma 2 LLM 结合**。如果能在 **PaliGemma** 这样的环境中**用 SigLIP 2 替换 SigLIP**，并观察模型的表现，那将会非常有趣！


