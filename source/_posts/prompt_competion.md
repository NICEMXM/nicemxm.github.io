---
title: GPT-4提示工程师竞赛-冠军心得
date: 2024-05-16
author: mxm
categories:
  - gpt
tags:
  - prompt
---

去年 11 月 8 日，新加坡政府科技局（GovTech）组织举办了首届 GPT-4 提示工程（Prompt Engineering）竞赛。数据科学家 Sheila Teo
最终夺冠，成为最终的提示女王（Prompt Queen）。之后，Teo 发布了一篇题为《我如何赢得了新加坡 GPT-4
提示工程赛》的博客文章，慷慨分享了其获胜法门。

博文原文：

上个月，我非常荣幸地赢得了新加坡首届 GPT-4 提示工程竞赛；该竞赛由新加坡政府科技局组织，汇聚了 400 多名优秀的参赛者。

提示工程是一门融合了艺术和科学的学科 —— 它既需要对技术的理解，也需要创造力和战略思维。这篇文章汇编了我一路以来学习到的提示工程策略，这些策略能让
LLM 切实完成你想完成的任务并做到更多！

本文包含以下内容，其中 🔵 是指适合初学者的提示工程技术，而 🔴 是指进阶技术。

1. [🔵] 使用 CO-STAR 框架来搭建 prompt 的结构
2. [🔵] 使用分隔符为 prompt 设置分节
3. [🔴] 使用 LLM 防护围栏创建系统 prompt
4. [🔴] 仅使用 LLM 分析数据集，不使用插件或代码 —— 附带一个实操示例：使用 GPT-4 分析一个真实的 Kaggle 数据集。

## 1. [🔵] 使用 CO-STAR 框架来搭建 prompt 的结构

为了让 LLM 给出最优响应，为 prompt 设置有效的结构至关重要。CO-STAR 框架是一种可以方便用于设计 prompt
结构的模板，这是新加坡政府科技局的数据科学与 AI 团队的创意成果。该模板考虑了会影响 LLM 响应的有效性和相关性的方方面面，从而有助于得到更优的响应。

其工作方式为：

![](CO-STAR.png)

(C) 上下文（Context）：提供与任务有关的背景信息。这有助于 LLM 理解正在讨论的具体场景，从而确保其响应是相关的。  
(O) 目标（Objective）：定义你希望 LLM 执行的任务。明晰目标有助于 LLM 将自己响应重点放在完成具体任务上。  
(S) 风格（Style）：指定你希望 LLM 使用的写作风格。这可能是一位具体名人的写作风格，也可以是某种职业专家（比如商业分析师或
CEO）的风格。这能引导 LLM 使用符合你需求的方式和词语给出响应。  
(T) 语气（Tone）：设定响应的态度。这能确保 LLM 的响应符合所需的情感或情绪上下文，比如正式、幽默、善解人意等。  
(A) 受众（Audience）：确定响应的目标受众。针对具体受众（比如领域专家、初学者、孩童）定制 LLM
的响应，确保其在你所需的上下文中是适当的和可被理解的。  
(R) 响应（Response）：提供响应的格式。这能确保 LLM 输出你的下游任务所需的格式，比如列表、JSON、专业报告等。对于大多数通过程序化方法将
LLM 响应用于下游任务的 LLM 应用而言，理想的输出格式是 JSON。

CO-STAR 的一个实际应用

这里通过一个实际案例说明 CO-STAR 为什么有用。假设你是一位社交媒体管理者，你需要帮助草拟一篇 Facebook 帖文，其內容是宣传你公司的新产品。

如果不使用 CO-STAR，那么你可能会使用这样的 prompt：

```chatinput
Write a facebook post to advertise my company’s new product. My company’s name is Alpha and the product is called Beta, a new ultra-fast hairdryer.
```

GPT-4 的输出如下：

```chatinput
ntroducing the Revolutionary Beta Hairdryer by Alpha!
- Say goodbye to long, tedious hair drying sessions! Alpha is proud to unveil the Beta, our latest innovation in haircare technology.
- Ultra-Fast Drying: The Beta is designed to cut your drying time in half, thanks to its advanced, powerful motor. Get ready faster and enjoy more time doing what you love!
Shop Now: [Link to product page]
```

这个结果也许够用了，但却只是泛泛而谈，缺乏具体的细节和有针对性的吸引力，而这些是吸引目标受众所必需的。

那如果使用 CO-STAR 模板呢？CO-STAR 会提醒你将任务涉及的其它方面也整合进 prompt 之中，具体来说就是上面的简单 prompt 中缺失的
STYLE、TONE、AUDIENCE 方面：

```chatinput
# CONTEXT #
I want to advertise my company’s new product. My company’s name is Alpha and the product is called Beta, which is a new ultra-fast hairdryer.
# OBJECTIVE #
Create a Facebook post for me, which aims to get people to click on the product link to purchase it.
# STYLE #
Follow the writing style of successful companies that advertise similar products, such as Dyson.
# TONE #
Persuasive
# AUDIENCE #
My company’s audience profile on Facebook is typically the older generation. Tailor your post to target what this audience typically looks out for in hair products.
# RESPONSE #
The Facebook post, kept concise yet impactful.
```

通过使用 CO-STAR，GPT-4 的响应会变得明显更有针对性、效果更好：

```chatinput
Embrace Effortless Elegance with Alpha Beta — The Ultra-Fast Hairdryer for the Wise Generation.
Rediscover the joy of simple, effective hair care with Alpha Beta. Our latest innovation is more than just a hairdryer; it’s a promise of swift, gentle, and reliable hair styling for those who appreciate the finer things in life.
- Easy and Intuitive Use: Say goodbye to complicated gadgets. Alpha Beta is
crafted for comfort and simplicity, perfect for those who value straightforward,
hassle-free technology.
- Time-Saving Technology: We understand your time is precious. That’s why Alpha Beta cuts down drying time significantly, giving you more moments to enjoy life’s pleasures.
Make every day a good hair day with Alpha Beta. Experience the blend of sophistication and simplicity today.
[Your Product Link Here]
```

CO-STAR 框架能引导用户以一种结构化的方式向 LLM 提供有关你的任务所有关键信息，从而确保得到完全符合你的需求的定制化的更优的响应。

## 2. [🔵] 使用分隔符为 prompt 设置分节

分隔符是一种特殊 token，可帮助 LLM 分辨 prompt 的哪些部分应被视为单个含义单元。这很重要，因为输入 LLM 的整个 prompt 是单个的
token 长序列。分隔符能将 prompt 中不同部分隔离开，从而为这个 token 序列提供结构，让其中各个部分能被区别对待。

需要说明的是，如果任务很简单，那么分隔符对 LLM 的响应质量的影响不大。但是，任务越复杂，使用分隔符分节对 LLM 响应的影响就越大。

### 用特殊字符当分隔符

分隔符可以使用任何通常不会同时出现的特殊字符序列，举些例子：###、===、>>>

特殊字符的数量和类型并不重要，只要它们足够独特即可，这样才能让 LLM 将它们理解成内容分隔符，而不是普通的标点符号。

下面通过一个例子说明如何在 prompt 中使用分隔符：

```chatinput
Classify the sentiment of each conversation in <<<CONVERSATIONS>>> as
‘Positive’ or ‘Negative’. Give the sentiment classifications without any other preamble text.

###
EXAMPLE CONVERSATIONS
[Agent]: Good morning, how can I assist you today?
[Customer]: This product is terrible, nothing like what was advertised!
[Customer]: I’m extremely disappointed and expect a full refund.
[Agent]: Good morning, how can I help you today?

[Customer]: Hi, I just wanted to say that I’m really impressed with your
product. It exceeded my expectations!
EXAMPLE OUTPUTS
Negative
Positive
###
<<<

[Agent]: Hello! Welcome to our support. How can I help you today?
[Customer]: Hi there! I just wanted to let you know I received my order, and
it’s fantastic!
[Agent]: That’s great to hear! We’re thrilled you’re happy with your purchase.
Is there anything else I can assist you with?

[Customer]: No, that’s it. Just wanted to give some positive feedback. Thanks
for your excellent service!
[Agent]: Hello, thank you for reaching out. How can I assist you today?
[Customer]: I’m very disappointed with my recent purchase. It’s not what I expected at all.
[Agent]: I’m sorry to hear that. Could you please provide more details so I can help?
[Customer]: The product is of poor quality and it arrived late. I’m really
unhappy with this experience.
>>>
```

上面例子中使用的分隔符是 ###，同时每一节都带有完全大写的标题以示区分，如 EXAMPLE CONVERSATIONS 和 EXAMPLE
OUTPUTS。前置说明部分陈述了要分类的对话是在 <<<CONVERSATIONS>>> 中，这些对话是在 prompt
末尾提供，也不带任何解释说明文本，但由于有了 <<< 和 >>> 这样的分隔符，LLM 就能理解这就是要分类的对话。

GPT-4 对此 prompt 给出的输出如下，其给出的情感分类结果不带任何附加文本，这符合我们的要求：

```chatinput
Positive
Negative
```

### 用 XML 标签当分隔符

另一种方法是使用 XML 标签作为分隔符。XML 标签是使用尖括号括起来的成对标签，包括开始和结束标签。比如 <tag> 和 </tag>。这很有效，因为
LLM 在训练时就看过了大量用 XML 标注的网络内容，已经学会了理解其格式。

下面用 XML 标签作为分隔符重写上面的 prompt：

```chatinput
Classify the sentiment of the following conversations into one of two classes, using the examples given. Give the sentiment classifications without any other
preamble text.

<classes>
Positive
Negative
</classes>

<example-conversations>
[Agent]: Good morning, how can I assist you today?
[Customer]: This product is terrible, nothing like what was advertised!
[Customer]: I’m extremely disappointed and expect a full refund.
[Agent]: Good morning, how can I help you today?

[Customer]: Hi, I just wanted to say that I’m really impressed with your
product. It exceeded my expectations!
</example-conversations>
<example-classes>

Negative
Positive
</example-classes>
<conversations>

[Agent]: Hello! Welcome to our support. How can I help you today?
[Customer]: Hi there! I just wanted to let you know I received my order, and
it’s fantastic!
[Agent]: That’s great to hear! We’re thrilled you’re happy with your purchase.
Is there anything else I can assist you with?

[Customer]: No, that’s it. Just wanted to give some positive feedback. Thanks
for your excellent service!
[Agent]: Hello, thank you for reaching out. How can I assist you today?
[Customer]: I’m very disappointed with my recent purchase. It’s not what I
expected at all.

[Agent]: I’m sorry to hear that. Could you please provide more details so I
can help?
[Customer]: The product is of poor quality and it arrived late. I’m really
unhappy with this experience.
</conversations>
```

为了达到更好的效果，在 XML 标签中使用的名词应该与指令中用于描述它们的名词一样。在上面的 prompt 中，我们给出的指令为：

```chatinput
Classify the sentiment of the following conversations into one of two classes, using the examples given. Give the sentiment classifications without any other preamble text.

```

其中使用的名词有 conversations、classes 和 examples。也因此，后面的分隔 XML 标签就对应为 <conversations>、<classes>、<
example-conversations> 和 <example-classes>。这能确保 LLM 理解指令与 XML 标签的关联。

同样的，使用这样的分隔符能以清晰的结构化方式对 prompt 进行分节，从而确保 GPT-4 输出的内容就刚好是你想要的结果：

```chatinput
Positive
Negative
```

## 3. [🔴] 使用 LLM 防护围栏创建系统提示

在深入之前，需要指出这一节的内容仅适用于具有 System Prompt（系统提示）功能的 LLM，而本文其它章节的内容却适用于任意
LLM。当然，具有这一功能的最著名 LLM 是 ChatGPT，因此这一节将使用 ChatGPT 作为示例进行说明。

与 System Prompts 有关的术语

首先，我们先把术语搞清楚：对于 ChatGPT，有大量资源使用 System Prompts、System Messages 和 Custom Instructions
这三个术语，而且很多时候它们的意思似乎差不多。这给很多人（包括我）带来了困扰，以至于让 OpenAI 都专门发了一篇文章来解释这些它们。简单总结一下：

* System Prompts 和 System Messages 是通过 ChatGPT 的 Chat Completions API 以程序化方式使用该 LLM 时使用的术语。
* 另一方面，Custom Instructions 是通过 https://chat.openai.com/ 的用户界面使用 ChatGPT 时的术语。

不过整体而言，这三个术语指代的是同一对象，因此请不要过多纠结于此！我们这一节将使用 System Prompts 这个术语。现在继续深入吧！

### System Prompts 是什么？

System Prompts 是指附加的额外 prompt，其作用是指示 LLM 理应的行为方式。之所以说这是额外附加的，是因为它位于「普通」prompt（也被称为用户
prompt）之外。

在一组聊天中，每一次你都要提供一个新的 prompt，System Prompts 的作用就像是一个 LLM 会自动应用的过滤器。这意味着，在一组聊天中，LLM
每次响应都要考虑 System Prompts。

### 应在何时使用 System Prompts？

你脑袋冒出的第一个问题可能是：我为什么应该在 System Prompts 中提供指令，毕竟我可以在一组聊天的第一个 prompt 中提供这些指令？

答案是因为 LLM 的对话记忆有局限。如果在一组对话的第一个 prompt 中提供这些指令，随着对话的进行，LLM 可能会「遗忘」你提供的第一个
prompt，其中的指令也就失效了。

另一方面，如果在 System Prompts 中提供这些指令，那么 LLM 就会自动将其与新的 prompt 一起纳入考量。这能确保随着对话进行，LLM
能持续接收这些指令，无论聊天变得多长。

总结一下：使用 System Prompts 提供你希望 LLM 在整个聊天过程中全程记住的指令。

### System Prompts 应包含什么内容？

System Prompts 中的指令通常包含以下类别：

* 任务定义，这样 LLM 在聊天过程中能一直记得要做什么。
* 输出格式，这样 LLM 能一直记得自己应该如何响应。
* 防护围栏，这样 LLM 能一直记得自己不应该如何响应。防护围栏（Guardrails）是 LLM 治理方面一个新兴领域，是指为 LLM 配置的可运行操作的边界。

举个例子，System Prompt 可能是这样的：

```chatinput
You will answer questions using this text: [insert text].
You will respond with a JSON object in this format: {“Question”: “Answer”}.
If the text does not contain sufficient information to answer the question, do not make up information and give the answer as “NA”.

You are only allowed to answer questions related to [insert scope]. Never answer any questions related to demographic information such as age, gender, and religion.
```

其中每部分的类别如下：

```markdown
任务定义：You will answer questions using this text: [insert text].
输出格式：You will respond with a JSON object in this format: {“Question”: “Answer”}.
防护围栏（幻觉）：If the text does not contain sufficient information to answer the question, do not make up information
and give the answer as “NA”.
防护围栏（范围）：You are only allowed to answer questions related to [insert scope]. Never answer any questions related to
demographic information such as age, gender, and religion.
```

### 那么「普通」prompt 又该包含哪些内容呢？

现在你可能会想：看起来 System Prompt 中已经给出了大量信息。那么我们又该在「普通」prompt（也称为用户 prompt）中放什么内容？

System Prompt 会大致描述任务概况。在上面的 System Prompt 示例中，任务被定义为仅使用特定的文本进行问答，并指示 LLM 以 {"
Question": "Answer"} 的格式进行响应。

```chatinput
You will answer questions using this text: [insert text].
You will respond with a JSON object in this format: {“Question”: “Answer”}.
```

在这个案例中，聊天中的每个用户 prompt 都只是你希望得到文本解答的问题。举个例子，用户 prompt 可能是这样「What is the text
about?」。而 LLM 的响应会是这样：{"What is the text about?": "The text is about..."}。

但我们可以进一步泛化这个示例任务。在实践中，你更可能会有多个希望得到解答的问题，而不只是一个。在这个案例中，我们可以将上述
System Prompt 的第一行从

```chatinput
You will answer questions using this text: [insert text].
```

改成

```chatinput
You will answer questions using the provided text.
```

现在，每个用户 prompt 中都既包含执行问答所基于的文本，也包含所要回答的问题。

```chatinput
<text>
[insert text]
</text>
<question>
[insert question]
</question>
```

这里，我们依然使用 XML 标签作为分隔符，以一种结构化的方式为 LLM 提供这两段所需信息。此处 XML 标签中使用的名词是 text 和
question，对应于 System Prompt 中使用的名词，这样一来 LLM 就能理解这些标签与 System Prompt 指令有何关联。

总结起来，System Prompt 应能给出整体的任务指令，而每个用户 prompt 应提供你希望执行任务时使用的确切细节。比如在这个案例中，这个确切的细节是文本和问题。

### 另：让 LLM 防护围栏变得动态化

在上面，防护围栏是通过 System Prompt 中的几句话添加的。然后，这些防护围栏在聊天的整个过程中就不变了。那如果你希望在对话的不同位置使用不同的防护围栏呢？

不幸的是，对于 ChatGPT 用户界面的用户，目前还没有能做到这一点的简单方法。但是，如果你通过编程方法与 ChatGPT
交互，你就很幸运了！现在人们对构建有效的 LLM 防护围栏的兴趣越来越大，有研究者开发了一些开源软件包，可让用户能以编程方式设置远远更加细节和动态的防护围栏。

英伟达团队开发的 NeMo Guardrails 尤其值得注意，这能让用户配置与 LLM
之间的期望对话流，从而在聊天的不同位置设置不同的防护围栏，实现随聊天不断演进的动态防护围栏。我强烈建议你研究看看！

## 4. [🔴] 仅使用 LLM 分析数据集，不使用插件或代码

你可能听说过 OpenAI 为 GPT-4 版本的 ChatGPT 提供的 Advanced Data Analysis（高级数据分析）插件 —— 高级（付费用户）可以使用。这让用户可以向
ChatGPT 上传数据集，然后直接在数据集上运行代码，实现精准的数据分析。

但你知道吗，其实不使用这样的插件也能让 LLM 分析数据集？我们首先了解一下完全使用 LLM 分析数据集的优势和局限。

### LLM 不擅长的数据集分析类型

你可能已经知道，LLM 执行准确数学计算的能力有限，这使得它们不适合需要对数据集进行精确定量分析的任务，比如：

* 描述性统计数值计算：以定量方式总结数值列，使用的度量包括均值或方差。
* 相关性分析： 获得列之间的精确相关系数。
* 统计分析：比如假设测试，可以确定不同数据点分组之间是否存在统计学上的显著差异。
* 机器学习：在数据集上执行预测性建模，可以使用的方法包括线性回归、梯度提升树或神经网络。

正是为了在数据集上执行这样的定量分析任务，OpenAI 才做了 Advanced Data Analysis 插件，这样才能借助编程语言来为这些任务在数据集上执行代码。

那么，为什么还需要不使用插件、仅使用 LLM 来分析数据集呢？

### LLM 擅长的数据集分析类型

LLM 擅长识别模式和趋势。这种能力源自 LLM 训练时使用的大量多样化数据，这让它们可以识别出可能并不显而易见的复杂模式。

这让他们非常适合处理基于模式发现的任务，比如：

* 异常检测：基于一列或多列数值识别偏离正常模式的异常数据点。
* 聚类：基于列之间的相似特征对数据点进行分组。
* 跨列关系：识别列之间的综合趋势。
* 文本分析（针对基于文本的列）： 基于主题或情绪执行分类。
* 趋势分析（针对具有时间属性的数据集）：识别列之中随时间演进的模式、季节变化或趋势。

对于这些类型的基于模式的任务，实际上相比于使用代码，仅使用 LLM 可能还能在更短的时间内得到更好的结果。下面通过一个示例来完整演示一番。

仅使用 LLM 来分析 Kaggle 数据集

该示例会使用一个常用的真实世界 Kaggle 数据集，该数据集是为客户个性分析任务收集整理的，其中的任务目标是对客户群进行细分，以更好地了解客户。

为了方便后面验证 LLM 的分析结果，这里仅取用一个子集，其中包含 50 行和最相关的列。之后，用于分析的数据集如下所示，其中每一行都代表一个客户，列则描述了客户信息：

