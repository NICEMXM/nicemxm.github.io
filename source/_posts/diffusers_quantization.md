---
title: 探索Diffusers中的Quantization后端
date: 2025-06-16
author: mxm
categories:
  - Diffusers
tags:
  - generate
---

像 Flux（一种基于流的文本到图像生成模型）这样的大型扩散模型可以创建令人惊叹的图像，但它们的大小可能是一个障碍，需要大量的内存和计算资源。Quantization 提供了一个强大的解决方案，缩小了这些模型，使其更易于访问，而不会严重影响性能。但最大的问题始终是：您真的能分辨出最终图像中的差异吗？
在我们深入研究 Hugging Face Diffusers 中各种量化后端如何工作的技术细节之前，测试一下

## 探索量化模型
我们创建了一个设置，您可以在其中提供提示，并使用原始的高精度模型（例如，BF16 中的 Flux-dev）和几个量化版本（BnB 4 位、BnB 8 位）生成结果。然后将生成的图像呈现给您，您的挑战是确定哪些图像来自量化模型。

通常，尤其是对于 8 位量化，差异很细微，如果不仔细检查，可能不会察觉。更激进的量化（如 4 位或更低）可能更明显，但结果仍然很好，尤其是考虑到大量内存节省。不过，NF4 通常会给出最好的权衡。

现在，让我们更深入地了解。

## Diffusers 中的量化后端

在我们之前的博文 “使用 Quanto 和 Diffusers 的内存高效 Diffusion Transformers”的基础上，本文探讨了直接集成到 Hugging Face Differs 中的各种量化后端。我们将研究 bitsandbytes、GGUF、torchao、Quanto 和原生 FP8 支持如何使大型和强大的模型更易于访问，并演示它们与 Flux 的配合使用。

在深入研究量化后端之前，我们先介绍 FluxPipeline（使用 black-forest-labs/FLUX.1-dev 检查点）及其组件，我们将对其进行量化。以 BF16 精度加载完整模型大约需要 31.447 GB 内存。主要组成部分是：FLUX.1-dev

文本编码器（CLIP 和 T5）：

功能：处理输入文本提示。FLUX-dev 使用 CLIP 进行初步理解，使用更大的 T5 进行细致的理解和更好的文本渲染。

内存：T5 - 9.52 GB; CLIP - 246 MB (in BF16)

Transformer （主型号 - MMDiT）：

功能：核心生成部件 （Multimodal Diffusion Transformer）。从文本嵌入在潜在空间中生成图像。

内存：23.8 GB（在 BF16 中）

变分自动编码器 （VAE）：

功能：在像素空间和潜在空间之间平移图像。将生成的潜在表示解码为基于像素的图像。

内存：168 MB（在 BF16 中）

量化的重点：示例将主要集中在 和 （T5） 上，以实现最可观的内存节省。transformertext_encoder_2

```
prompts = [
    "Baroque style, a lavish palace interior with ornate gilded ceilings, intricate tapestries, and dramatic lighting over a grand staircase.",
    "Futurist style, a dynamic spaceport with sleek silver starships docked at angular platforms, surrounded by distant planets and glowing energy lines.",
    "Noir style, a shadowy alleyway with flickering street lamps and a solitary trench-coated figure, framed by rain-soaked cobblestones and darkened storefronts.",
]
```

### 位字节 （BnB）

BitSandbytes 是一种常用且用户友好的 8 位和 4 位量化库，广泛用于 LLM 和 QLoRA 微调。我们也可以将其用于基于 transformer 的扩散和流动模型。

![image](https://github.com/user-attachments/assets/1a604a92-bd77-4756-ba26-857dbcffff2a)

| 精度 |	加载后的内存 |	峰值内存 |	推理时间 |
|-----|------|-------|--------|
| BF16 |	~31.447 GB |	36.166 GB |	12 秒 |
|4 位 |	12.584 GB |	17.281 GB |	12 秒 |
| 8 位 |	19.273 GB |	24.432 GB |	27 秒 |

所有基准测试均在 1 个 NVIDIA H100 80GB GPU 上执行

示例（BnB 4 位的 Flux-dev）：

```
import torch
from diffusers import FluxPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

model_id = "black-forest-labs/FLUX.1-dev"

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
        "text_encoder_2": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
    }
)

pipe = FluxPipeline.from_pretrained(
    model_id,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

prompt = "Baroque style, a lavish palace interior with ornate gilded ceilings, intricate tapestries, and dramatic lighting over a grand staircase."
pipe_kwargs = {
    "prompt": prompt,
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 50,
    "max_sequence_length": 512,
}


print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")

image = pipe(
    **pipe_kwargs, generator=torch.manual_seed(0),
).images[0]

print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")

image.save("flux-dev_bnb_4bit.png")

```

### torchao
torchao 是一个用于架构优化的 PyTorch 原生库，提供量化、稀疏性和自定义数据类型，旨在与 FSDP 兼容。Diffusers 支持各种 的 奇异数据类型，从而支持对模型优化的精细控制。torch.compiletorchao

![image](https://github.com/user-attachments/assets/3b128a0b-449c-4f82-ae3f-6fc864f6910e)

| 精度 |	加载后的内存 |	峰值内存 |	推理时间 |
|-----|------|-------|--------|
| int4_weight_only |	10.635 GB |	14.654 GB |	109 秒 |
| int8_weight_only |	12.584 GB |	21.482 GB |	15 秒 |
| float8_weight_only |	17.016 GB |	21.488 GB |	15 秒 |

示例（仅带有 torchao INT8 权重的 Flux-dev）：

```
@@
- from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
+ from diffusers import TorchAoConfig as DiffusersTorchAoConfig

- from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
+ from transformers import TorchAoConfig as TransformersTorchAoConfig
@@
pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
-         "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
-         "text_encoder_2": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
+         "transformer": DiffusersTorchAoConfig("int8_weight_only"),
+         "text_encoder_2": TransformersTorchAoConfig("int8_weight_only"),
    }
)

```

示例（仅带有 torchao INT4 权重的 flux-dev）：

```
@@
- from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
+ from diffusers import TorchAoConfig as DiffusersTorchAoConfig

- from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
+ from transformers import TorchAoConfig as TransformersTorchAoConfig
@@
pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
-         "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
-         "text_encoder_2": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
+         "transformer": DiffusersTorchAoConfig("int4_weight_only"),
+         "text_encoder_2": TransformersTorchAoConfig("int4_weight_only"),
    }
)

pipe = FluxPipeline.from_pretrained(
    model_id,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
+    device_map="balanced"
)
- pipe.to("cuda")

```

### Quanto

Quanto 是一个量化库，通过optimum库与 Hugging Face 生态系统集成。

![image](https://github.com/user-attachments/assets/00d47111-1cba-457e-8bd4-3f12f0a1c078)

| quanto精度 |	加载后的内存 |	峰值内存 |	推理时间 |
|-----|------|-------|--------|
|INT4 |	12.254  GB |	16.139 GB |	109 秒 |
|INT8 |	17.330 GB |	21.814 GB |	15 秒 |
| FP8 |	16.395 GB |	20.898 GB |	16 秒 |

### GGUF

GGUF 是 llama.cpp 社区中流行的一种文件格式，用于存储量化模型。

![image](https://github.com/user-attachments/assets/1ccaff5d-7936-414b-84f6-9dba8df20ab9)


| GGUF精度 |	加载后的内存 |	峰值内存 |	推理时间 |
|-----|------|-------|--------|
| Q2_k |	13.264 GB |	17.752 GB |	26 秒 |
| Q4_1 |	16.838 GB |	21.326 GB |	23 秒 |
| Q8_0 |	21.502 GB |	25.973 GB |	15 秒 |

示例（使用 GGUF Q4_1的 Flux-dev）

```
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

model_id = "black-forest-labs/FLUX.1-dev"

# Path to a pre-quantized GGUF file
ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_1.gguf"

transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

pipe = FluxPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

```

### FP8 Layerwise Casting 

FP8 Layerwise Casting 是一种内存优化技术。它的工作原理是将模型的权重存储为紧凑的 FP8（8 位浮点）格式，该格式使用的内存大约是标准 FP16 或 BF16 精度的一半。 在层执行计算之前，其权重会动态转换为更高的计算精度（如 FP16/BF16）。紧接着，权重被投射回 FP8 以实现高效存储。这种方法之所以有效，是因为核心计算保持了高精度，并且通常会跳过对量化（如归一化）特别敏感的层。此技术还可以与组卸载结合使用，以进一步节省内存。

![image](https://github.com/user-attachments/assets/5e90a772-a146-4c7b-aa41-c4294d735e93)

| 精度 |	加载后的内存 |	峰值内存 |	推理时间 |
|-----|------|-------|--------|
| FP8 （e4m3） |	23.682 GB |	28.451 GB |	13 秒 |

```
import torch
from diffusers import AutoModel, FluxPipeline

model_id = "black-forest-labs/FLUX.1-dev"

transformer = AutoModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

pipe = FluxPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda")

```

### 结合 More Memory Optimizations 和 torch.compile

这些量化后端中的大多数都可以与 Diffusers 中提供的内存优化技术结合使用。让我们来探讨一下 CPU 卸载、组卸载和 .您可以在 Diffusers 文档中了解有关这些技术的更多信息。torch.compile

示例（BnB 4 位 + enable_model_cpu_offload的 Flux-dev）：

```
import torch
from diffusers import FluxPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

model_id = "black-forest-labs/FLUX.1-dev"

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
        "text_encoder_2": TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
    }
)

pipe = FluxPipeline.from_pretrained(
    model_id,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16
)
- pipe.to("cuda")
+ pipe.enable_model_cpu_offload()

```

模型 CPU 卸载 （enable_model_cpu_offload）：此方法在推理管道期间在 CPU 和 GPU 之间移动整个模型组件（如 UNet、文本编码器或 VAE）。它可以节省大量 VRAM，并且通常比更精细的卸载更快，因为它涉及更少、更大的数据传输。

| 精度 |	加载后的内存 |	峰值内存 |	推理时间 |
|-----|------|-------|--------|
| 4 位 |	12.383 GB |	12.383 GB |	17 秒 |
| 8 位 |	19.182  GB |	23.428 GB |	27 秒 |

示例（使用 fp8 分层转换 + 组卸载的 Flux-dev）：

```
import torch
from diffusers import FluxPipeline, AutoModel

model_id = "black-forest-labs/FLUX.1-dev"

transformer = AutoModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    # device_map="cuda"
)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
+ transformer.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), offload_type="leaf_level", use_stream=True)

pipe = FluxPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)
- pipe.to("cuda")

```

组卸载（enable_group_offload 用于 diffusers 组件或 apply_group_offloading 用于通用 torch.nn.Modules）：它将内部模型层组（如 or 实例）移动到 CPU。这种方法通常比完整模型卸载更节省内存，并且比顺序卸载更快。torch.nn.ModuleListtorch.nn.Sequential

### FP8 Layerwise Casting + 组卸载：

| 精度 |	加载后的内存 |	峰值内存 |	推理时间 |
|-----|------|-------|--------|
| FP8 （e4m3） |	9.264 GB |	14.232 GB |	58 秒 |

示例（使用 torchao 4 位 + torch.compile 的 Flux-dev）：

```
import torch
from diffusers import FluxPipeline
from diffusers import TorchAoConfig as DiffusersTorchAoConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import TorchAoConfig as TransformersTorchAoConfig

from torchao.quantization import Float8WeightOnlyConfig

model_id = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer":DiffusersTorchAoConfig("int4_weight_only"),
        "text_encoder_2": TransformersTorchAoConfig("int4_weight_only"),
    }
)

pipe = FluxPipeline.from_pretrained(
    model_id,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="balanced"
)

+ pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)

```

torch.compile：另一种补充方法是使用 PyTorch 2.x 的 torch.compile（） 功能加速模型的执行。编译模型不会直接降低内存，但可以显著加快推理速度。PyTorch 2.0 的编译 （Torch Dynamo） 的工作原理是提前跟踪和优化模型图形。

torchao + torch.compile 中：

| torchao 精度 |	加载后的内存 |	峰值内存 |	推理时间 | 编译时 |
|-----|------|-------|--------|--------|
| int4_weight_only |	10.635 GB |	15.238 GB |	6 秒 | ~285 秒|
| int8_weight_only |	17.020 GB |	22.473 GB |	8 秒 | ~851 |
| float8_weight_only |	17.016 GB |	22.115 GB |	8 秒 | ~545 |

## 结论
以下是选择量化后端的快速指南：

最容易节省内存 （NVIDIA）：从 4/8 位开始。这也可以结合使用以加快推理速度。bitsandbytestorch.compile()

优先考虑推理速度: 可以用于潜在地提高推理速度。torchao GGUF bitsandbytes torch.compile()

对于硬件灵活性 （CPU/MPS），FP8 精度：可能是一个不错的选择。Quanto

简单性 （Hopper/Ada）：探索 FP8 Layerwise Casting().enable_layerwise_casting

对于使用现有的 GGUF 模型：使用 GGUF loading().from_single_file

量化显著降低了使用大型扩散模型的准入门槛。尝试使用这些后端，以找到满足您需求的最佳内存、速度和质量平衡。
