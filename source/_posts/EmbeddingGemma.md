---
title: EmbeddingGemma
date: 2025-09-15
author: mxm
categories:
  - LLM
tags:
  - generate
---

## 引言

Google 发布了 EmbeddingGemma，这是一种最先进的多语言嵌入模型，非常适合设备上的用例。该模型专为速度和效率而设计，具有 308M 参数的紧凑尺寸和 2K 上下文窗口，为移动 RAG 管道、代理等释放了新的可能性。EmbeddingGemma 经过训练，支持 100 多种语言，在撰写本文时，它是海量文本嵌入基准 （MTEB） 上排名最高的 500M 以下纯文本多语言嵌入模型。

文本嵌入已成为现代自然语言应用程序的支柱，将单词、句子和文档转换为捕获含义、情感和意图的密集向量。这些向量支持跨海量语料库的快速相似性搜索、聚类、分类和检索，为从推荐引擎和语义搜索到检索增强生成和代码搜索工具的所有工具提供支持。计算这些嵌入的嵌入模型被广泛使用，Hugging Face 的每月下载量超过 2 亿次。

在此基础上，Google DeepMind 的 EmbeddingGemma 成为迄今为止最新、功能最强大的小型多语言嵌入模型。EmbeddingGemma 仅具有 308M 参数、2k 令牌上下文窗口和对 100 多种语言的支持，可在大规模多语言文本嵌入基准测试 （MMTEB） 上提供最先进的性能，同时在量化时保持在 200 MB 以下的 RAM。

各种设计选择造就了一个非常实用的开源工具，用于在日常设备上计算高质量的多语言嵌入。

在这篇博文中，我们描述了 EmbeddingGemma 架构和训练，并向您展示了如何将该模型与各种框架（如 Sentence Transformers、LangChain、LlamaIndex、Haystack、txtai、Transformers.js、文本嵌入推理和 ONNX）一起使用。

之后，我们将演示如何在您的域上微调 EmbeddingGemma 以获得更强大的性能。在我们的示例中，我们在医疗指令和检索数据集 （MIRIAD） 上微调了 EmbeddingGemma。由此产生的模型 sentence-transformers/embeddinggemma-300m-medical 在我们的任务上实现了最先进的性能：检索科学医学论文的段落以回答详细的医学问题。在这项任务上，它甚至比两倍大的模型更胜一筹。

## 架构

EmbeddingGemma 建立在 Gemma3 transformer 主干之上，但经过修改以使用双向注意力而不是因果（单向）注意力。这意味着序列中较早的标记可以关注较晚的标记，从而有效地将架构从解码器转变为编码器。编码器模型在检索等嵌入任务上的性能可以优于解码器LLM（Weller 等人，2025 年）。有了这个主干，模型可以一次处理相当大的 2048 个标记，足以满足典型的检索输入，特别是考虑到较大的输入通常会导致文本嵌入中的信息丢失。

除了生成标记嵌入的新基于 Gemma3 的编码器主干之外，均值池层将这些标记嵌入转换为文本嵌入。最后，两个密集的层将文本嵌入转换为其最终形式，即 768 维向量。

EmbeddingGemma 模型已使用Matryoshka Representation Learning （MRL） 进行训练，允许您根据需要将 768 维输出截断为 512、256 或 128 维。这导致更快的下游处理并降低内存和磁盘空间利用率。请参阅 Sentence Transformers 用法，了解显示如何执行此截断的片段。

该模型已使用精心策划的多语言语料库进行训练，总计约 3200 亿个代币。专有数据集混合了公开可用的 Web 文本、代码和技术文档以及特定于合成任务的示例。它经过过滤，以避免儿童性虐待材料 （CSAM）、敏感数据以及低质量或不安全的内容。

## 评估

EmbeddingGemma 以 MMTEB（多语言，v2）和 MTEB（英语，v2）套件为基准，它们涵盖广泛的任务、领域和语言。尽管参数大小适中，但该模型始终优于同类基线，同时保持非常小的内存占用。

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/62e67d7d-4404-4233-a68b-d5153a9d5be2" />

结果将列在官方 MTEB 排行榜上。我们排除了任何在超过 20% 的 MTEB 数据上训练过的模型，以减轻潜在的过度拟合。

## 用法

EmbeddingGemma 与许多流行的工具集成，可以轻松整合到您现有的工作流程和应用程序中。该模型已集成到 Sentence Transformers 中，因此也集成在幕后使用 Sentence Transformers 的项目中，例如 LangChain、LlamaIndex、Haystack 和 txtai。请参阅以下示例，开始使用您喜欢的框架。

对于生产部署，您可以使用文本嵌入推理 （TEI） 在各种硬件配置上有效地为模型提供服务，并且可以使用Transformers.js用于 Web 应用程序。

无论您选择何种框架，您都应该注意提示。对于嵌入模型，提示会附加到输入文本前面，以允许模型区分不同的任务。EmbeddingGemma 使用这些提示名称和提示进行了训练，因此在使用模型时也应包含它们：

```
query: ,"task: search result | query: "
document: ,"title: none | text: "
BitextMining: ,"task: search result | query: "
Clustering: ,"task: clustering | query: "
Classification: ,"task: classification | query: "
InstructionRetrieval: ,"task: code retrieval | query: "
MultilabelClassification: ,"task: classification | query: "
PairClassification: ,"task: sentence similarity | query: "
Reranking: ,"task: search result | query: "
Retrieval-query: ,"task: search result | query: "
Retrieval-document: ,"title: none | text: "
STS: ,"task: sentence similarity | query: "
Summarization:"task: summarization | query: "
```

在 Sentence Transformers 中，调用 和 时会自动使用 和 提示，但对于其他框架，您可能必须： $querydocumentmodel.encode_querymodel.encode_document

* 指定提示名称（例如“重新排名”），
* 指定提示字符串（例如“任务：搜索结果 |query： “），或
* 手动将提示添加到输入文本前面。

## Sentence Transformers

您需要安装以下软件包：

```
pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
pip install sentence-transformers>=5.0.0
```

### 检索
使用句子转换器进行推理相当简单，请参阅以下语义搜索示例：

```
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")

# Run inference with queries and documents
query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
query_embeddings = model.encode_query(query)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# (768,) (4, 768)

# Compute similarities to determine a ranking
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.3011, 0.6359, 0.4930, 0.4889]])

# Convert similarities to a ranking
ranking = similarities.argsort(descending=True)[0]
print(ranking)
# tensor([1, 2, 3, 0])
```

 ### LangChain

 如果您愿意，还可以使用 LangChain ，它在幕后使用 Sentence Transformers。请注意，您必须告诉 LangChain 分别使用名为“query”和“document”的提示来处理查询和文档。此示例涉及简单的信息检索设置，但相同的嵌入模型可用于更复杂的方案。HuggingFaceEmbeddings

您需要安装以下软件包：

```
pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
pip install sentence-transformers
pip install langchain
pip install langchain-community
pip install langchain-huggingface
pip install faiss-cpu
```

```
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Download the model from the 🤗 Hub. Also specify to use the "query" and "document" prompts
# as defined in the model configuration, as LangChain doesn't automatically use them.
# See https://huggingface.co/google/embeddinggemma-300m/blob/main/config_sentence_transformers.json
embedder = HuggingFaceEmbeddings(
    model_name="google/embeddinggemma-300m",
    query_encode_kwargs={"prompt_name": "query"},
    encode_kwargs={"prompt_name": "document"}
)

data = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

# Create documents for the vector store
documents = [Document(page_content=text, metadata={"id": i}) for i, text in enumerate(data)]

# Create vector store using FAISS. Setting distance_strategy to "MAX_INNER_PRODUCT" uses
# FAISS' FlatIndexIP behind the scenes, which is optimized for inner product search. This
# is what the model was trained for
vector_store = FAISS.from_documents(documents, embedder, distance_strategy="MAX_INNER_PRODUCT")

# Search for top 3 similar documents
query = "Which planet is known as the Red Planet?"
results = vector_store.similarity_search_with_score(query, k=3)

# Print results
for doc, score in results:
    print(f"Text: {doc.page_content} (score: {score:.4f})")
"""
Text: Mars, known for its reddish appearance, is often referred to as the Red Planet. (score: 0.6359)
Text: Jupiter, the largest planet in our solar system, has a prominent red spot. (score: 0.4930)
Text: Saturn, famous for its rings, is sometimes mistaken for the Red Planet. (score: 0.4889)
"""

```

### LlamaIndex

LlamaIndex 也支持 EmbeddingGemma，因为它在后台使用句子转换器。为了获得正确的行为，您需要指定模型配置中定义的查询和文档提示。否则，您的表现将不理想。此脚本显示了将 EmbeddingGemma 与 LlamaIndex 一起使用的基本示例，但您也可以在更困难的设置中使用该类。HuggingFaceEmbedding

您需要安装以下软件包：

```
pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
pip install sentence-transformers
pip install llama-index
pip install llama-index-embeddings-huggingface
pip install llama-index-vector-stores-faiss

```

```
import faiss
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

# Download from the 🤗 Hub. Also specify the query and document prompts as
# defined in the model configuration, as LlamaIndex doesn't automatically load them.
# See https://huggingface.co/google/embeddinggemma-300m/blob/main/config_sentence_transformers.json
embeddings = HuggingFaceEmbedding(
    model_name="google/embeddinggemma-300m",
    query_instruction="task: search result | query: ",
    text_instruction="title: none | text: ",
)

data = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

# Create a sample vector store
store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(768))
store.add([TextNode(id=i, text=text, embedding=embeddings.get_text_embedding(text)) for i, text in enumerate(data)])

# Search for top k similar documents
query = "Which planet is known as the Red Planet?"
query_embedding = embeddings.get_query_embedding(query)
results = store.query(VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=3))

# Print results
for idx, score in zip(results.ids, results.similarities):
    print(f"Text: {data[int(idx)]} (score: {score:.4f})")
"""
Text: Mars, known for its reddish appearance, is often referred to as the Red Planet. (score: 0.6359)
Text: Jupiter, the largest planet in our solar system, has a prominent red spot. (score: 0.4930)
Text: Saturn, famous for its rings, is sometimes mistaken for the Red Planet. (score: 0.4889)
"""

```

## 微调

与 Sentence Transformers 库兼容的所有模型一样，EmbeddingGemma 可以在您的特定数据集上轻松微调。为了展示这一点，我们将对医学指导和检索数据集 （MIRIAD） 数据集进行微调，以便我们的微调模型变得特别擅长从给定详细医学问题的科学医学论文中查找多达 1000 个标记的段落。这些段落可以用作生成模型更有效地回答问题的关键上下文。google/embeddinggemma-300m

下面，您可以使用可展开的选项卡探索微调过程的每个关键组件。每个选项卡都包含相关代码和详细说明。

### 模型

```
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData

model = SentenceTransformer(
    "google/embeddinggemma-300m",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="EmbeddingGemma-300m trained on the Medical Instruction and RetrIeval Dataset (MIRIAD)",
    ),
)
# SentenceTransformer(
#   (0): Transformer({'max_seq_length': 1024, 'do_lower_case': False, 'architecture': 'Gemma3TextModel'})
#   (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
#   (2): Dense({'in_features': 768, 'out_features': 3072, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})
#   (3): Dense({'in_features': 3072, 'out_features': 768, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})
#   (4): Normalize()
# )

```
此代码从 Hugging Face 加载 EmbeddingGemma 模型，并带有可选的模型卡元数据，用于文档和共享。该类加载模型权重和配置，而参数附加有助于包含在自动生成的模型卡中的元数据。SentenceTransformermodel_card_data

### 数据

```
from datasets import load_dataset

train_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="train").select(range(100_000))
eval_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="eval").select(range(1_000))
test_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="test").select(range(1_000))
# Dataset({
#     features: ['question', 'passage_text'],
#     num_rows: 100000
# })
# Dataset({
#     features: ['question', 'passage_text'],
#     num_rows: 1000
# })
# Dataset({
#     features: ['question', 'passage_text'],
#     num_rows: 1000
# })

```

此代码加载 MIRIAD 数据集，或者更确切地说，加载已分为训练、评估和测试拆分的副本。使用大型、高质量的数据集可确保模型学习有意义的表示，而子集可以加快实验速度。该函数从 Hugging Face Datasets 中获取数据集，并且该方法限制每次拆分的样本数。load_dataset.select()

### 损失函数

```
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)
```
此代码使用缓存多负排名损失 （CMNRL） 定义训练的损失函数。CMNRL 对于检索任务很有效，因为它使用批量内负值来有效地训练模型以区分正确和不正确的对。损失采用问答对并将批次中的其他答案视为负数，从而最大化嵌入空间中不相关对之间的距离。该参数控制内存使用，但不影响训练动态。mini_batch_size

### 训练参数

```
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainingArguments

run_name = "embeddinggemma-300m-medical-100k"
args = SentenceTransformerTrainingArguments(
    output_dir=f"models/{run_name}",
    num_train_epochs=1,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if your GPU can't run FP16
    bf16=False,  # Set to True if your GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    prompts={
        "question": model.prompts["query"],
        "passage_text": model.prompts["document"],
    },
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=20,
    run_name=run_name,
)

```

此代码设置了用于训练、评估和日志记录的所有超参数和配置。正确的训练论证对于高效、稳定和可重复的训练至关重要。参数控制批量大小、学习率、混合精度、评估和保存频率等。值得注意的是，该字典将数据集列映射到模型用于区分查询和文档的提示。prompts

### 评估

```
from sentence_transformers.evaluation import InformationRetrievalEvaluator

queries = dict(enumerate(eval_dataset["question"]))
corpus = dict(enumerate(eval_dataset["passage_text"] + train_dataset["passage_text"][:30_000]))
relevant_docs = {idx: [idx] for idx in queries}
dev_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="miriad-eval-1kq-31kd",
    show_progress_bar=True,
)
dev_evaluator(model)

```

此代码设置了一个用于信息检索的评估器，使用查询和语料库来衡量模型性能。训练期间的评估有助于监控进度并避免过度拟合。评估器通过检查模型是否检索到每个查询的正确段落来计算检索指标（NDCG、MRR、召回率、精度、MAP 等）。它可以在训练之前、期间和之后运行，结果将被记录并合并到自动生成的模型卡中。

请注意，此片段特别使用所有 （1k） 评估问题，以对照所有 （1k） 评估段落和 30k 训练段落的语料库，总共 31k 文档。对于模型来说，仅根据评估段落进行评估太简单了。

### 训练

```
from sentence_transformers import SentenceTransformerTrainer

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()
```

### 完整的微调脚本

下面是完整的脚本，结合了上述所有组件：

```
import logging
import traceback

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "google/embeddinggemma-300m",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="EmbeddingGemma-300m trained on the Medical Instruction and RetrIeval Dataset (MIRIAD)",
    ),
)

# 3. Load a dataset to finetune on
train_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="train").select(range(100_000))
eval_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="eval").select(range(1_000))
test_dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="test").select(range(1_000))

# 4. Define a loss function. CachedMultipleNegativesRankingLoss (CMNRL) is a special variant of MNRL (a.k.a. InfoNCE),
# which take question-answer pairs (or triplets, etc.) as input. It will take answers from other questions in the batch
# as wrong answers, reducing the distance between the question and the true answer while increasing the distance to the
# wrong answers, in the embedding space.
# The (C)MNRL losses benefit from larger `per_device_train_batch_size` in the Training Arguments, as they can leverage
# more in-batch negative samples. At the same time, the `mini_batch_size` does not affect training performance, but it
# does limit the memory usage. A good trick is setting a high `per_device_train_batch_size` while keeping
# `mini_batch_size` small.
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)

# 5. (Optional) Specify training arguments
run_name = "embeddinggemma-300m-medical-100k"
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # (Cached)MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    prompts={  # Map training column names to model prompts
        "question": model.prompts["query"],
        "passage_text": model.prompts["document"],
    },
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=20,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator using the evaluation queries and 31k answers & evaluate the base model
queries = dict(enumerate(eval_dataset["question"]))
corpus = dict(enumerate(eval_dataset["passage_text"] + train_dataset["passage_text"][:30_000]))
relevant_docs = {idx: [idx] for idx in queries}
dev_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="miriad-eval-1kq-31kd",  # 1k questions, 31k passages
    show_progress_bar=True,
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# (Optional) Evaluate the trained model on the evaluation set once more, this will also log the results
# and include them in the model card
dev_evaluator(model)

queries = dict(enumerate(test_dataset["question"]))
corpus = dict(enumerate(test_dataset["passage_text"] + train_dataset["passage_text"][:30_000]))
relevant_docs = {idx: [idx] for idx in queries}
test_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="miriad-test-1kq-31kd",  # 1k questions, 31k passages
    show_progress_bar=True,
)
test_evaluator(model)

# 8. Save the trained model
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)

# 9. (Optional) Push it to the Hugging Face Hub
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
try:
    model.push_to_hub(run_name)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{run_name}')`."
    )

```
我们在配备 3090GB VRAM 的 RTX 24 上运行了完整的训练脚本，完成的训练和评估脚本耗时 5.5 小时。如果需要，您可以通过减少实例上的 和 来进一步减少内存占用。

