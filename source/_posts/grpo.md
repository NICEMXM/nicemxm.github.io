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

**计算优势**

对于每个G序列中，我们使用 reward 模型计算 Reward。为了与奖励模型的比较性质保持一致（通常在同一问题的输出比较数据集上进行训练），计算Reward以反映这些相对比较。它按如下方式规范化：

$$ \hat{A}_{i,t} = \frac{r_i - \text{mean}(r)}{\text{std}(r)} $$

**估计KL散度**

KL散度是通过Schulman等人（2020）提出的近似器来估计的。近似器的定义如下： 

$$D_{KL} [\pi_{\theta} \parallel \pi_{ref}] = \frac{\pi_{\theta}(o_{i,t} \mid q,o_{i,<t})}{\pi_{ref}(o_{i,t} \mid q,o_{i,<t})} - \log \left( \frac{\pi_{\theta}(o_{i,t} \mid q,o_{i,<t})}{\pi_{ref}(o_{i,t} \mid q,o_{i,<t})} \right) - 1,$$
 
**计算损失**
目标是最大化优势，同时确保模型保持接近参考策略。因此，损失定义如下：
 
$$L_{GRPO}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \left[ \frac{\pi_\theta(o_{i,t}|\mathbf{q}, o_{i,<t})}{[\pi_\theta(o_{i,t}|\mathbf{q}, o_{i,<t})]_{\text{no grad}}} \hat{A}_{i,t} - \beta D_{KL}[\pi_\theta \| \pi_{\text{ref}}] \right],$$
 
 其中第一项代表缩放后的优势，第二项通过KL散度惩罚与参考策略的偏差。
 
 在原始论文中，为了在每次生成后进行多次更新，利用了裁剪代理目标来推广这种表述方式：
 
$$
L_{GRPO}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \left[\min\left(\frac{\pi_\theta(o_{i,t}|\mathbf{q}, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|\mathbf{q}, o_{i,<t})} \hat{A}_{i,t}, \text{clip}\left(\frac{\pi_\theta(o_{i,t}|\mathbf{q}, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|\mathbf{q}, o_{i,<t})}, 1-\epsilon, 1+\epsilon\right) \hat{A}_{i,t}\right) - \beta D_{KL}[\pi_\theta \| \pi_{\text{ref}}]\right],
$$

这里的 $\text{clip}(\cdot, 1-\epsilon, 1+\epsilon)$ 确保更新不会过度偏离参考策略，通过将策略比率限制在 $(1-\epsilon)$ 到 $(1+\epsilon)$ 之间。然而，在TRL中，如同原始论文一样，我们每次生成只做一次更新，因此可以简化损失至第一种形式。

总的来说，组相对策略优化 （GRPO） 是一种强化学习技术，旨在通过利用基于组的奖励和策略优化来微调语言模型。它建立在近端策略优化 （PPO） 的概念之上，但通过考虑组内生成输出的相对性能，引入了一种新的奖励计算和策略更新方法。

## 使用GRPO微调模型

使用 GRPO 微调 SmolLM 模型涉及根据推理、准确性和格式等关键因素优化从奖励得出的替代损失。微调过程遵循以下步骤：
* 安装所需的软件包
* 加载和测试基本模型
* 定义 Helper 函数
* 定义奖励函数
* 设置 GRPO 配置和训练器
* 准备 GSM8K 数据集
* 配置 Trainer 和 Model
* 训练模型
* 使用 Fine-Tuned 模型进行推理

### 第 1 步：安装所需的库

首先，安装必要的软件包。我们正在从其 GitHub 存储库安装最新版本的 Hugging Face Transformers、Accelerate、Datasets 和 TRL 库。我们还安装了 PEFT 以实现参数高效的微调。

```python
!pip install -q git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/accelerate.git
!pip install -q datasets huggingface-hub trl
!pip install -q git+https://github.com/huggingface/peft.git
#!pip install flash-attn --no-build-isolation
```

### 第 2 步：导入库并定义辅助函数
导入所有必需的库并定义用于解析和格式化模型输出的辅助函数。这些函数将有助于从模型的 XML 格式响应中提取答案。

```python
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfiga  # Note: This might be a typo. In your code, you later call LoraConfig.
from trl import GRPOConfig, GRPOTrainer

# System prompt that instructs the model to use a specific XML format.
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# XML chain-of-thought format template.
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# Function to extract the answer part from the XML response.
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Function to extract an answer if it is provided with a "####" delimiter.
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()
```

### 第 3 步：准备 GSM8K 数据集
我们使用来自 Hugging Face Hub 的 GSM8K 数据集（小学数学题的集合）。在该函数中，我们将每个示例转换为带有系统说明和用户问题的提示。（一次性示例已被注释掉，但如果需要，可以启用。get_gsm8k_questions

```python
# Function to load and process the GSM8K dataset.
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]  # Load the GSM8K dataset.
    data = data.map(lambda x: {  # Process each example.
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            # Uncomment the following lines to include a one-shot example.
            # {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            # {'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            # )},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

# Load the processed dataset.
dataset = get_gsm8k_questions()
```

### 第 4 步：定义奖励函数
定义了几个奖励函数来指导训练过程。这些函数评估模型输出的不同方面，例如正确性、格式和对 XML 格式的结构遵守性。

```python
# Reward function to check correctness: compares the extracted answer from the response with the known answer.
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# Reward function that checks if the response is a digit.
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# Reward function that checks if the response strictly follows the desired XML format.
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# Reward function with a softer check for the XML format.
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in responses]

# Function to count specific XML tokens and award a small reward for each.
def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

# Reward function that uses the XML token count.
def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
```

### 第 5 步：设置模型和 Tokenizer
我们从 Hugging Face Hub 中选择 SmolLM 模型 （）。模型将加载数据类型并移动到 GPU。此外，还将加载 tokenizer，并将其 padding token设置为序列结束 token。HuggingFaceTB/SmolLM2-135M-Instructbfloat16

```python
# Choose the model name.
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
# Alternatively, you can use:
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Set output directories and run name based on the chosen model.
if "SmolLM2" in model_name:
    output_dir = "outputs/SmolLM2-135M-GRPO"
    run_name = "SmolLM2-135M-GRPO"
else:
    output_dir = "outputs/Qwen-1.5B-GRPO"
    run_name = "Qwen-1.5B-GRPO-gsm8k"

# Load the model.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    #attn_implementation="flash_attention_2",
    device_map=None
).to("cuda")

# Load the tokenizer and ensure that the pad token is set.
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

### 第 6 步：配置 GRPO 和 PEFT
接下来，我们定义 GRPO 的训练配置以及使用 LoRA （Low-Rank Adaptation） 的 PEFT （Parameter-Efficient Fine-Tuning） 配置。请注意，在下面的代码中，PEFT 配置已创建，但未传递给 trainer（它已被注释掉）。您可以通过取消注释相应的参数来启用它。

```python
# GRPO training configuration.
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=16,  # Must be divisible by num_generations.
    gradient_accumulation_steps=4,
    num_generations=16,  # Number of generations per prompt.
    max_prompt_length=256,
    max_completion_length=786,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="none",
    log_on_each_node=False,
)

# PEFT configuration using LoRA.
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)
```

注意：在导入部分，我们使用了 this 可能是拼写错误。确保从 中正确导入和使用 。LoraConfigaLoraConfigpeft

### 第 7 步：初始化 GRPO 训练器
现在，我们使用模型、分词器（作为 传递）、奖励函数、训练配置和数据集来实例化 GRPOTrainer。奖励函数应用于每一代，以提供精细的反馈。processing_class

```python
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func
    ],
    args=training_args,
    train_dataset=dataset,
    # peft_config=peft_config  # Uncomment this line to enable LoRA-based parameter-efficient fine-tuning.
)
```

### 第 8 步：开始训练过程
最后，在训练程序上调用该方法，开始使用 GRPO 微调模型。train()

```python
trainer.train()
```

### 第 9 步：使用 Fine-Tuned 模型进行推理
训练后，在新问题上测试微调后的模型。

```python
sample = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "If there are 12 cookies in a dozen and you have 5 dozen, how many cookies do you have?"}
]
final_prompt = tokenizer.apply_chat_template(sample, tokenize=False)
print(inference(final_prompt, os.path.join(config.output_dir, "final_model")))
```
