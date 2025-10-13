---
title: 可视化 VLM 的工作原理
date: 2025-10-13
author: mxm
categories:
  - LLM
tags:
  - VLM
---

## 介绍
视觉语言模型 （VLM） 是自回归 AI 模型，可将文本和图像处理为输入。在这篇文章中，我们将仔细研究 Idefics3 和 SmolVLM 等 VLM 如何在幕后运行，探索它们如何合并视觉和文本信息以生成连贯的输出。

作为本博文中的模型参考，我们将使用 HuggingFaceTB/SmolVLM-256M-Instruct。

## 处理器
### 图像处理器
处理器将文本和图像数据准备成适合模型的统一格式。对于图像，它会执行一系列转换，然后再将它们转换为模型可以理解的类似标记的表示。

图像管道可以可视化如下：

<img width="1012" height="606" alt="image" src="https://github.com/user-attachments/assets/4f016172-4caa-4060-a75c-2aa8de239444" />

此过程中的一个关键步骤是图像分割，如此代码片段所示。 每个图像都分为更小的补丁（或“拆分”），这些补丁是单独编码的。

当文本伴随图像时，每个图像首先由文本序列中的标记表示。根据拆分次数，每个图像都会扩展为：<image>64×number_of_splits

常数 64 来自以下关系：

512^2/16^2 / 4^2 = 64

我们稍后会回到这个公式，但现在，将其视为：

-- 每个图像分割由 64 个标记表示。

### 文本处理器
包含图像时，文本还必须使用占位符反映它。处理器遵循以下步骤：<image>

* 插入占位符 – 输入中的每个图像都由文本中的标记表示。<image>
* 计数分割 – 处理器确定每个图像在预处理后具有多少分割。
* 扩展令牌 – 然后根据以下公式，根据图像分割的数量将每个令牌替换（或“扩展”）为一个序列。<image>

<img width="4380" height="1173" alt="image" src="https://github.com/user-attachments/assets/5608afcd-f9d2-44fc-b536-ac5315ddb60d" />

要完全掌握图像标记的扩展方式，请查看下面的编码示例并查看以下参考：

* 文档示例 – 显示了扩展如何发生的实际示例。
* Main 函数 – 定义扩展逻辑的核心实现。

编码示例
```
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

# Load images
image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")

out = processor.decode(inputs["input_ids"][0])

print(out.replace("<image>", ".")) # printing the full <image> will bload the screen

```

## 数据准备
在输入模型之前，图像和文本数据都经过准备和对齐。 与大多数自回归模型一样，输出序列向右移动，以教模型如何根据所有先前的标记预测下一个标记。

但是，由于无法直接预测图像表示，因此使用标记对其进行屏蔽。 这可以防止模型计算非文本（视觉）标记的损失。<pad>

<img width="1533" height="413" alt="image" src="https://github.com/user-attachments/assets/5a4ca169-138b-4b2a-8022-455fc080d7df" />

## 模型架构
### 嵌入层
从高层次来看，SmolVLM 由五个主要组件组成（如右图所示）。 文本处理从提示开始，提示被标记化并通过嵌入层传递。该层将离散标记转换为高维向量表示，产生形状张量：

[sequence_length,576]

<img width="2861" height="3065" alt="image" src="https://github.com/user-attachments/assets/c445d14c-2c38-4a6a-b3f1-53edcbad0bf7" />

该张量现在对输入文本的向量表示进行编码，并作为所有后续计算的基础，请注意，该张量已经为即将处理的图像提供了 13 x 64 个占位符标记（其中 13 是图像分割的数量）

<img width="811" height="487" alt="image" src="https://github.com/user-attachments/assets/f653b50f-9bdd-457d-8b26-6390f27e1fa5" />

### 视觉模型
#### 补丁嵌入

对于视觉分支，输入张量的形状为 。 例如，对于可以拆分为 13 个 RGB 拆分的图像，图像表示为 。[splits, num_channels, height, width][13, 3, 512, 512]

该图像被送入 patch embedding 层后，会被转换为一系列视觉 token，从而与 Transformer 架构兼容。具体做法是将图像划分为多个小的、不重叠的图像块（patches），并将每个图像块投影到一个高维向量空间中。其核心原理是将 RGB 通道视为嵌入维度，并将其从 3 维扩展到 768 维。

该操作通过二维卷积实现，其中卷积核大小（kernel_size）和步长（stride）均被设置为图像块（patch）的尺寸。这样可确保每个卷积窗口恰好处理一个图像块，且彼此之间无重叠。kernel_sizestride

```
self.patch_embedding = nn.Conv2d(
    in_channels=config.num_channels,  # 3 (RGB channels)
    out_channels=self.embed_dim,      # 768
    kernel_size=self.patch_size,      # 16
    stride=self.patch_size,           # 16
    padding="valid",
)
```

应用时，转换将按如下方式进行：[13, 3, 512, 512]

|步	|操作	|输出形状	|描述|
|---|-----|---------|----|
|1	|转换2d	|[13, 768, 32, 32]	|每个 16×16 补丁都嵌入到 768 维向量 （） 中。512 / 16 = 32|
|2	|重塑	|[13, 1024, 768]	|32×32 网格被扁平化为 1024 个补丁（可视标记）。|

生成的张量将每个图像表示为 1024 个嵌入补丁的序列，准备与 Transformer 中的文本嵌入一起处理。[13, 1024, 768]

#### 位置编码器

位置编码器注入有关补丁顺序和空间布局的信息，就像位置编码对文本模型中的单词所做的那样。 由于转换器没有固有的秩序感，因此这些编码允许模型了解每个补丁在图像中的位置。

#### 编码器

编码器以简单的方式运行：

* 多头注意力 （MHA） 层捕获图像块之间的关系，使模型能够推理空间依赖性。
* 然后输出通过前馈网络 （MLP） 来细化和投影特征。
  
这种注意力实现接近 BERT 的功能，必须注意的是，vision_model的输出是给定我们正在使用的示例输入的。[13,1024,768]

#### 连接器
<img width="1911" height="912" alt="image" src="https://github.com/user-attachments/assets/56179a70-71b9-4da5-8689-9bc8bc906de9" />

连接器充当视觉编码器和语言模型之间的桥梁，确保两种模式共享兼容的嵌入空间。 它的两个主要功能是：

* 压缩视觉输出（减少标记数）。
* 转换嵌入维度以匹配文本嵌入维度。

#### 像素重排
Pixel Shuffle（像素重排）操作在压缩视觉特征空间维度的同时，保留了重要的空间关系。具体而言，它将图像 token 的数量从 1024 减少到 64，大幅降低了序列长度，同时保持了特征的表达丰富性。

该变换过程如下所示：

| 描述 | 代码|
|------|-----|
|  # (split height and width) <br> → [splits, H, W, C] <br> #  (apply transformation on width dimension)  <br> →  [splits, H, W/scale, C×scale] <br> #  (permute dimensions)  <br> → [splits, W/scale, H, C×scale]  <br> #  (apply transformation on height dimension) <br> → [splits, W/scale, H/scale, C×scale²] <br> # (permute dimensions back) <br> → [splits, H/scale, W/scale, C×scale²] <br> # (merge height and width back) <br> → [splits, (H/scale)×(W/scale), C×scale²] <br>    |  <img width="3680" height="1256" alt="image" src="https://github.com/user-attachments/assets/adb7e406-f36c-4241-811f-1d42f4d0ea17" />|

通过这种方式，我们得到的最终维度为：  

\[
[\text{splits},\ H' / \text{scale} \times W' / \text{scale},\ C \times \text{scale}^2]
\]  

即：  

\[
[13,\ 1024 / 4^2,\ 768 \times 4^2] \rightarrow [13,\ 64,\ 12288]
\]

通过逐步对像素进行重排（shuffle）和重组，该操作在压缩高度和宽度维度的同时，保留了每个方向上视觉信息的空间顺序。最终得到一个更加紧凑的张量，同时保留了图像的关键特征。

#### 模态投影（Modality Projection）

该连接器还应用了一个模态投影层（modality_projection layer），该层是一个线性变换，用于将视觉特征的维度调整为与文本 token 的嵌入维度一致。通过这一操作，张量维度从  

\[
[13,\ 64,\ 12288]
\]  

变换为  

\[
[13,\ 64,\ 576]
\] 。

### 输入合并
输入合并是一个不可学习的层，实际上是一个负责将视觉和文本嵌入集成到单个输入序列中的函数。 它将文本序列中的每个<图像>占位符标记替换为连接器生成的相应视觉嵌入。

实际上，此函数会扫描令牌 ID 以查找令牌，并用预处理的图像表示形式替换它们。此步骤有效地将两种模态合并为一个可以直接输入解码器的连续张量。<image>

### 解码器
解码器的功能类似于传统的自回归语言模型。它由堆叠的屏蔽多头注意力 （MHA） 层和语言建模 （LM） 头组成。

* 屏蔽 MHA 确保每个标记只能关注以前的标记（包括视觉标记），从而在文本生成过程中保留因果关系。
* LM 头将解码器的隐藏状态映射回词汇 logits，使模型能够预测下一个标记。
  
这使得 VLM 能够生成连贯的多模态输出，将文本预测无缝地建立在视觉环境中。

这里要记住的关键是，由于我们无法计算图像标记的输出，因此我们在目标中使用pad标记来跳过计算它的损失。

<img width="6907" height="5157" alt="image" src="https://github.com/user-attachments/assets/9e4fc295-609b-4992-a413-74e1993a1282" />

## 结论
在这篇文章中，我们探讨了 SmolVLM 等视觉语言模型 （VLM） 的内部工作原理，详细分析了它们如何处理和集成从原始像素和文本到连贯、接地输出的多模态数据。

以下是每个阶段的快速回顾：

* 处理器： 准备和对齐原始文本和图像输入。
* 视觉模块：将像素数据转换为高维补丁嵌入。
* 连接器： 将视觉特征压缩并投影到与文本标记相同的嵌入空间中。
* 输入合并：用视觉嵌入替换占位符标记，形成统一的多模态序列。
* 解码器： 通过关注视觉和文本信息来生成上下文感知文本。
  
从本质上讲，VLM 不仅看和读，还跨模态推理。这种架构允许它们处理多个图像、纯文本提示，甚至纯图像输入，使 SmolVLM 成为各种多模态应用程序的灵活而强大的基础。
