---
title: TRL ⚡️ 中的视觉语言模型对齐
date: 2025-08-11
author: mxm
categories:
  - LLM
tags:
  - generate
---
## 引言

视觉语言模型 （VLM） 变得越来越强大，但使其与人类偏好保持一致仍然很重要。在 TRL 中，我们已经展示了如何使用监督微调 （SFT） 和直接偏好优化 （DPO） 对 VLM 进行后期训练。这一次，我们走得更远。

我们在TRL中增加了两种新的多模态对齐方法：群体相对策略优化（GRPO）、其变体群体序列策略优化（GSPO）和混合偏好优化（MPO）。所有这些都让您超越成对 DPO，从首选项数据中提取更多信号，并使用现代 VLM 更好地扩展。

## 视觉语言模型的对齐
传统上，您会采用基本模型，应用 SFT 以遵循说明，然后应用 DPO 将其与优先数据保持一致。之前，我们将这种方法应用于视觉语言模型 （VLM），并在 IDEFICS2 上进行了验证，显示模型响应有所改善。

DPO 的工作原理是使用对比损失来优化模型响应对之间的偏好：您有一个选择的答案和一个被拒绝的答案，您可以根据您想要和不想要的内容优化您的偏好。

但在去年，新的多模态对齐方法（GRPO 和 MPO）越来越受欢迎，可以进一步推动 VLM 性能。在博客文章的末尾，您可以找到一个表格，其中展示了模型响应之间的差异。

### 混合偏好优化 （MPO）
由于分布偏移，将多模态模型与 SFT 对齐以执行推理任务是失败的。同时，与 DPO 一致的模型无法产生连贯的基本原理，并且可能会产生重复的响应。为了解决这个问题，有一种称为混合偏好优化 （MPO） 的新技术，专门用于多模态模型。该方法本质上是DPO的扩展，具有多种损失：DPO（sigmoid）的偏好损失、二元分类器优化（BCO）的质量损失和SFT的生成损失。根据该论文，只需切换到这种综合损失即可使 MathVista 提高 6.2 个百分点！

<img width="3337" height="2256" alt="image" src="https://github.com/user-attachments/assets/e1045b60-e498-4874-b0cf-c68e4f7dc2d4" />

由于这只是修改损失，因此我们在 TRL 的类中添加了组合损失支持。要使用它，您可以按如下方式初始化：DPOTrainerDPOConfig

```
mpo_config = DPOConfig(
    output_dir=tmp_dir,
    per_device_train_batch_size=2,
    learning_rate=9e-1,
    loss_type=["sigmoid", "bco_pair", "sft"], # Loss types to combine, as used in the MPO paper
    loss_weights=[0.8, 0.2, 1.0], # Corresponding weights, as used in the MPO paper
    report_to="none",
    bf16=False,
    fp16=False,
    use_cpu=True,
    max_steps=1,
)

```

然后初始化 ：DPOTrainer

```
mpo_trainer = DPOTrainer(
    model=model_id,
    args=mpo_config,
    processing_class=tokenizer,
    train_dataset=dataset,
)
mpo_trainer.train()

```

就是这样！如果您想进一步探索，可以在此处(https://huggingface.co/learn/cookbook/fine_tuning_vlm_mpo)找到完整的笔记本示例。

### 多模态组相对策略优化（GRPO）

组相对策略优化（GRPO）是一种尖端的对齐方法，最初在DeepSeek数学论文中引入，后来集成到开创性的LLMDeepSeek R1中。这是对 PPO 的补充，其中政策更新是通过组（代表对话如何展开的轨迹批次）完成的。此功能使奖励噪声更加稳健，因为噪声在组内平均。由于模型学习的是更广泛意义上的良好响应，而不是单一的高奖励样本，因此这种方法也使模型具有高性能。

<img width="3525" height="1861" alt="image" src="https://github.com/user-attachments/assets/3701d0e0-aecb-48a2-9b37-fd9aa04ccabd" />

在 TRL 中，我们现在引入了对视觉语言模型的 GRPO 支持。你可以在notebook中找到它。相反，我们将重点强调关键组件和概念。

为了使训练脚本有效工作，我们需要验证答案的格式是否正确，并且解决方案本身是否接近已完成的部分，因此我们编写了两个奖励函数。为了真正看到后一种奖励的改进，您需要一个相当极端主义的设置，其中您拥有相对更大的模型、大量代数和高质量、多样化的数据集。

```
import re
from math_verify import LatexExtractionConfig, parse, verify

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    matches = [re.match(pattern, content) for content in completions]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    rewards = [1.0 if match else 0.0 for match in matches]
    print(completions)
    print(rewards)
    return rewards

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['solution']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards

```

然后，您可以初始化 GRPOConfig 和 GRPOTrainer，传入我们上面定义的奖励函数并调用 train（） 开始训练。

```
from trl import GRPOConfig

training_args = GRPOConfig(
    learning_rate=1e-5,
    remove_unused_columns=False,
    max_prompt_length=None,
    .. # setup other params of choice here
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
    processing_class=processor
)
trainer.train()
```

在此处 (https://huggingface.co/learn/cookbook/fine_tuning_vlm_grpo_trl) 浏览完整的笔记本示例。

### 组序列策略优化 （GSPO）

组序列策略优化（Group Sequence Policy Optimization，GSPO）是Qwen最近发布的一种RL对齐算法，它克服了GRPO的一些局限性。它实现了更稳定的训练计算重要性，在序列级别而不是每个标记上抽样权重。它的好处在 MoE 风格模型中更为相关。

最新的 TRL 还引入了对 GSPO 的支持，由于它是 GRPO 损失的变体，因此它提供了多模式支持。要创建训练器，过程与 GRPO 相同，但添加了以下额外参数（值是从论文中提取的）。

```
from trl import GRPOConfig

training_args = GRPOConfig(
        ...
    importance_sampling_level="sequence",
    epsilon=3e-4,
    epsilon_high=4e-4,
    beta=0.0,
    loss_type="grpo",
    steps_per_generation=1,
    steps_per_generation=4
)

```

比较
下表总结了使用上述技术微调的 Qwen2.5VL-3B 的模型输出。请注意，我们对数据集子集进行了最少的运行，并且模型在不同的数据集上进行了微调，因此进行了比较。

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e2895616-22c0-4e78-9562-ce1a73d5bd66" />

```
问题：在给定的图中，如果角度 APO 为 25.0 度，则角度 AOB 的度量是多少？选择： A： 140° B： 130° C： 120° D： 110°

数据集中的答案：要找到角度 AOB 的度量，我们可以使用四边形内角之和等于 360 度的性质。应用此属性，我们可以将角度 AOB 计算为 360 - 角度 OAP - 角度 OBP - 角度 P。由于角度 OAP 和角度 OBP 都是 90 度，并且角度 P 为 50 度，因此我们可以将这些值代入等式中，得到角度 AOB = 360 - 90 - 90 - 50 = 130 度。因此，角度 AOB 的测量值为 130 度。因此，答案是选项B。答案：B

基本模型输出：要找到角度 AOB 的度量，我们需要使用圆中相同弧线所赋予的角度的属性。具体来说，圆心处的弧线所折腾的角度（角 AOB）是圆周上任意点处相同弧线所折磨的角度（角 APO）的两倍。假设角度 APO = 25°，我们可以按如下方式计算角度 AOB：由于角度 AOB 是角度 APO 的两倍，因此我们有：[ 角度 AOB = 2 * 角度 APO ][ 角度 AOB = 2 * 25° ][ 角度 AOB = 50° ]然而，根据提供的选择，这种计算似乎不正确。让我们使用正确的属性重新评估。正确的属性指出，圆心处的圆弧所折磨的角度是圆周上任意点上同一圆弧所折磨的角度的两倍。因此，正确的计算应该是： 角度 AOB = 2 * 角度 APO = 2 * 25° = 50°但由于这与给定的选择不匹配，让我们考虑另一种方法。如果角 APO 为 25°，并且考虑到循环四边形的性质或循环四边形中相反角度之和为 180° 的事实，我们可能需要查看圆弧和弦形成的补充角。让我们假设正确的解释涉及找到给定角度 APO 的补充角度，这将为我们提供角度 AOB 的度量。角度 AOB = 180° - 角度 APO = 180° - 25° = 155°这仍然与选项不匹配。鉴于提供的选择，似乎可能存在误解或需要不同的解释。但是，根据标准几何原理，正确答案应该是： 角度 AOB = 2 * 角度 APO = 2 * 25° = 50° 如果不是这种情况，请提供额外的上下文或澄清问题陈述。

MPO 输出：要找到角度 AOB 的度量，我们需要使用圆中相同弧线所赋予的角度的属性。具体来说，圆心处的弧线所折腾的角度（角 AOB）是圆周上任意点处相同弧线所折磨的角度（角 APO）的两倍。假设角度 APO = 25°，我们可以按如下方式计算角度 AOB： 由于角度 AOB 是角度 APO 的两倍，因此我们有：[ 角度 AOB = 2 * 角度 APO ][ 角度 AOB = 2 * 25° ][ 角度 AOB = 50° ]然而，根据提供的选择，这种计算似乎不正确。让我们使用正确的属性重新评估。正确的属性指出，圆心处的圆弧所折磨的角度是圆周上任意点上同一圆弧所折磨的角度的两倍。因此，正确的计算应该是： 角度 AOB = 2 * 角度 APO = 2 * 25° = 50°但由于这与给定的选择不匹配，让我们考虑另一种方法。如果我们看三角形 OAP，角 AOP 是 90°，因为它是直角（因为它内接在半圆中）。现在，角度 APO 给出为 25°，因此角度 AOA 将为 65°（因为它是角度 APO 的补充）。因此，角度 AOB 将是它的一半，即 32.5°。然而，这也与选择不符。让我们考虑一下问题可能涉及不同的解释或不同的几何配置的可能性。给定选择，让我们假设问题陈述或选择本身可能存在错误。如果我们要从给定的选项中进行选择，那么根据标准几何原理，对于此类问题，它们似乎都没有一个正确拟合。

GRPO 输出：要找到角度 AOB 的度量，我们需要了解角度 AOB 是一个内接角，它截取与中心角 AOP 相同的弧线。根据内接角定理，内接角的度量是同一弧的中心角度量的一半。给定：- 角度 APO = 25.0 度由于角 AOB 是截取与中心角 AOP 相同弧线的内接角，因此我们可以使用内接角定理：角 AOB 的度量是角 AOP 度量的一半。因此，角度 AOB 的测量值为 130 度。

GSPO 输出：要找到角 AOB 的度量，我们需要了解角 AOB 是一个内接角，它截取与中心角 AOP 相同的弧线。根据内接角定理，内接角的度量是同一弧的中心角度量的一半。给定：- 角度 APO = 25.0 度由于角 AOB 是截距与中心角 AOP 相同弧的内接角，因此我们可以使用内接角定理来找到角 AOB 的度量：角 AOB 的度量是角 AOP 度量的一半。因此，答案是 B：130°。
```

## TRL 中的 vLLM 集成
vLLM 集成在 TRL 中，以支持需要在训练期间生成样本的在线对齐方法。运行如下所示的示例脚本可启用 vLLM：

```
CUDA_VISIBLE_DEVICES=1,2 python3 examples/scripts/grpo_vlm.py     --model_name_or_path   Qwen/Qwen2.5-VL-3B-Instruct    …   --log_completions --use_vllm --vllm_mode colocate

```

主要有两种模式：和 .colocate 在与训练循环相同的进程中运行 vLLM，在训练和生成之间共享相同的 GPU，在 .同时，要求您在可以访问服务器的不同进程中单独提供 vLLM。您可以使用以下命令启动此服务器：colocateserverGRPOTrainerserver

```
trl vllm-serve --model Qwen/Qwen2.5-VL-3B-Instruct --tensor-parallel-size 1

```

然后，您可以按如下方式运行脚本。

```
CUDA_VISIBLE_DEVICES=1,2 python3 examples/scripts/grpo_vlm.py     --model_name_or_path   Qwen/Qwen2.5-VL-3B-Instruct    …   --log_completions --use_vllm --vllm_mode server

```

还有一个提示：我们添加了对在 TRL 中将 vLLM 与 Transformers 后端一起使用的支持。可以在运行具有 colocate 的脚本时或通过传递标志为模型提供服务时启用它。--vllm_model_impl transformers

您可以在此处(https://huggingface.co/docs/trl/en/vllm_integration)阅读有关 TRL 中 vLLM 集成的更多信息。

