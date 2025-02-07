
深度学习模型已经彻底改变了人工智能领域，但其庞大的规模和计算需求可能成为实际应用的瓶颈。模型蒸馏是一种强大的技术，它通过将知识从大型复杂模型（教师）转移到更小、更高效的模型（学生）来解决这一挑战。

在这篇博客中，我们将介绍如何使用 LoRA（低秩适应）等专业技术，将 DeepSeek-R1 的推理能力提炼成一个更小的模型，比如 Microsoft 的 Phi-3-Mini。

什么是蒸馏？
蒸馏是一种机器学习技术，其中较小的模型（“学生”）经过训练，以模仿较大的预训练模型（“老师”）的行为。目标是保留教师的大部分性能，同时显著降低计算成本和内存占用。

这个想法最早是在 Geoffrey Hinton 关于知识提炼的开创性论文中提出的。它不是直接在原始数据上训练学生模型，而是从教师模型的输出或中间表示中学习。这实际上是受到人类教育的启发。

看点：

成本效益：较小的模型需要的计算资源更少。
速度：非常适合延迟敏感型应用程序（例如 API、边缘设备）。
专业化：为特定领域定制模型，而无需重新培训巨头。
蒸馏的类型
有几种建模蒸馏方法，每种方法都有自己的优点：

1. 数据蒸馏

在数据蒸馏中，教师模型生成合成数据或伪标签，然后用于训练学生模型。
这种方法可以应用于广泛的任务，甚至是那些 logits 信息量较少的任务（例如，开放式推理任务）。
2. Logits 蒸馏

Logit 是神经网络在应用 softmax 函数之前的原始输出分数。
在 logits 蒸馏中，学生模型被训练为匹配教师的 logits，而不仅仅是最终预测。
这种方法保留了有关教师信心水平和决策过程的更多信息。
3. 特征蒸馏

特征蒸馏涉及将知识从教师模型的中间层传递给学生。
通过对齐两个模型的隐藏表示，学生可以学习更丰富、更抽象的特征。
Deepseek 的提炼模型
为了实现访问的民主化，DeepSeek AI 发布了六种基于流行架构的提炼变体，如 Qwen （Qwen， 2024b） 和 Llama （AI@Meta， 2024）。 他们使用通过 DeepSeek-R1 策划的 800k 样本直接微调了开源模型。

尽管比 DeepSeek-R1 小得多，但蒸馏模型在各种基准测试中都表现出令人印象深刻的性能，通常与更大的模型的功能相当甚至超过。如下图所示


Deepseek Distilled 模型基准测试 （https://arxiv.org/html/2501.12948v1)
为什么要 Distill 你自己的模型？
1. 特定任务的优化

预先提炼的模型在广泛的数据集上进行训练，可以在各种任务中表现良好。但是，实际应用程序通常需要专业化。

示例场景：

您正在构建一个财务预测聊天机器人。

在这种情况下，使用 DeepSeek-R1 为金融数据集（例如，股票价格预测、风险分析）生成推理跟踪，并将这些知识提炼成一个已经了解金融细微差别的较小模型（例如：金融-LLM）。

2. 大规模成本效益

虽然预先提取的模型很有效，但对于您的特定工作负载来说，它们可能仍然有点矫枉过正。通过提取自己的模型，您可以针对确切的资源约束进行优化。

3. 基准性能 ≠ 实际性能

预先提炼的模型在基准测试中表现出色，但基准测试通常不能代表实际任务。因此，您经常需要一个模型，该模型在实际场景中的性能比任何预先提炼的模型都要好。

4. 迭代改进

预先提炼的模型是静态的 — 它们不会随着时间的推移而改进。通过提取自己的模型，您可以在新数据可用时不断优化它。

代码教程：蒸馏 DeepSeek-R1 知识融入定制更小的模型
步骤 1：安装库
pip install -q torch transformers datasets accelerate bitsandbytes flash-attn --no-build-isolation
第 2 步：生成数据集并设置其格式
您可以通过使用 ollama 或任何其他部署框架在您的环境中部署 deepseek-r1 来生成自定义域相关数据集。但是，在本教程中，我们将使用 Magpie-Reasoning-V2 数据集，其中包含由 DeepSeek-R1 生成的 250K 思维链 （CoT） 推理样本。这些示例涵盖各种任务，如数学推理、编码和一般问题解决。

数据集结构

每个样本包括：

Instruction：任务描述（例如，“Solve this math question”）。
响应： DeepSeek-R1 的分步推理 （CoT）。
例：

{
  "instruction": "Solve for x: 2x + 5 = 15",
  "response": "<think>First, subtract 5 from both sides: 2x = 10. Then, divide by 2: x = 5.</think>"
}
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B", token="YOUR_HF_TOKEN")
dataset = dataset["train"]

# Format the dataset
def format_instruction(example):
    return {
        "text": (
            "<|user|>\n"
            f"{example['instruction']}\n"
            "<|end|>\n"
            "<|assistant|>\n"
            f"{example['response']}\n"
            "<|end|>"
        )
    }

formatted_dataset = dataset.map(format_instruction, batched=False, remove_columns=subset_dataset.column_names)
formatted_dataset = formatted_dataset.train_test_split(test_size=0.1)  # 90-10 train-test split
将数据集构建为 Phi-3 的聊天模板格式：

<|user|>：标记用户查询的开始。

<|assistant|>：标记模型响应的开始。

<|end|>：标记转弯的结束。

每个 LLM 都使用特定的格式来完成指令遵循任务。将数据集与此结构保持一致可确保模型学习正确的对话模式。因此，请确保根据要提取的模型设置数据格式。

第 3 步：加载模型和 Tokenizer
将特殊令牌 和 添加到 tokenizer。<think></think>

为了增强模型的推理能力，我们引入了这些。

<think>：标志着推理的开始。

</think>：标志着推理的结束。

这些标记可帮助模型学习生成结构化的分步解决方案。

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Add custom tokens
CUSTOM_TOKENS = ["<think>", "</think>"]
tokenizer.add_special_tokens({"additional_special_tokens": CUSTOM_TOKENS})
tokenizer.pad_token = tokenizer.eos_token

# Load model with flash attention
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
model.resize_token_embeddings(len(tokenizer))  # Resize for custom tokens
第 4 步：配置 LoRA 以实现高效的微调
LoRA 通过冻结基本模型并仅训练小的适配器层来减少内存使用量。

from peft import LoraConfig

peft_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.2,  # Dropout rate
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention layers
    bias="none",  # No bias terms
    task_type="CAUSAL_LM"  # Task type
)
第 5 步：设置训练参数
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./phi-3-deepseek-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=2e-5,
    fp16=True,
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine"
)
第 6 步：训练模型
SFTTrainer 简化了指令跟随模型的监督微调。这些批处理示例，并启用基于 LoRA 的训练。data_collatorpeft_config

from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
    data_collator=data_collator,
    peft_config=peft_config
)

# Start training
trainer.train()
trainer.save_model("./phi-3-deepseek-finetuned")
tokenizer.save_pretrained("./phi-3-deepseek-finetuned")
第 7 步：合并并保存最终模型
训练后，LoRA 适配器必须与基础模型合并以进行推理。此步骤可确保模型可以在没有 PEFT 的情况下独立使用。

final_model = trainer.model.merge_and_unload()
final_model.save_pretrained("./phi-3-deepseek-finetuned-final")
tokenizer.save_pretrained("./phi-3-deepseek-finetuned-final")
第 8 步：推理
from transformers import pipeline

# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained(
    "./phi-3-deepseek-finetuned-final",
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("./phi-3-deepseek-finetuned-final")
model.resize_token_embeddings(len(tokenizer))

# Create chat pipeline
chat_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Generate response
prompt = """<|user|>
What's the probability of rolling a 7 with two dice?
<|end|>
<|assistant|>
"""

output = chat_pipeline(
    prompt,
    max_new_tokens=5000,
    temperature=0.7,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)

print(output[0]['generated_text'])
下面您可以看到 phi 模型在蒸馏前后的响应。

问题：用两个骰子掷出 7 的概率是多少？

蒸馏前的推理：回答简单明了。它直接提供了计算答案的步骤。


蒸馏前的 Phi 推断
蒸馏后的推理： Distilled Response 引入了一种更详细和结构化的方法，包括一个明确的 “思考 ”部分，概述了思考过程和推理，这将对为复杂问题生成准确的响应非常有帮助。


蒸馏后的 Phi 推断
最后，蒸馏的模型被推送到 huggingface 集线器
